/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#define cublasCheck(ans) checkCublasStatus((ans), __FILE__, __LINE__)
static void checkCublasStatus(cublasStatus_t status, const char *file, int line) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS API failed with status %d %s %d\n", status, file, line);
		exit(1);
	}
}

template <class ElementA,
					class ElementB,
					class SmemLayoutA,
					class SmemLayoutB>
struct SharedStorage
{
	cute::array<ElementA, cute::cosize_v<SmemLayoutA>> A;
	cute::array<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
					class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
					class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
					class TC, class CStride, class CSmemLayout, class TiledMma,
					class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device_s2r(ProblemShape shape_MNK, CtaTiler cta_tiler,
						TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
						TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
						TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
						Alpha alpha, Beta beta)
{
	using namespace cute;

	// Preconditions
	CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
	CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

	CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
	CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

	static_assert(is_static<ASmemLayout>::value);
	static_assert(is_static<BSmemLayout>::value);
	static_assert(is_static<CSmemLayout>::value);

	CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
	CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
	CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
	CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
	CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
	CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

	CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
	CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
	CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

	//
	// Full and Tiled Tensors
	//

	// Represent the full tensors
	Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
	Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
	Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

	// Get the appropriate blocks for this thread block
	auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
	Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
	Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
	Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

	// Shared memory buffers
	extern __shared__ char shared_memory[];
	using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
	SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
	Tensor sA = make_tensor(make_smem_ptr(smem.A.data()), sA_layout);    // (BLK_M,BLK_K,PIPE)
	Tensor sB = make_tensor(make_smem_ptr(smem.B.data()), sB_layout);    // (BLK_N,BLK_K,PIPE)

	//
	// Partition the copying of A and B tiles across the threads
	//

	ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
	Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
	Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

	ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
	Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
	Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

	CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
	CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
	CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
	CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

	//
	// PREFETCH
	//

	auto K_PIPE_MAX = size<3>(tAsA);

	// Total count of tiles
	int k_tile_count = size<3>(tAgA);
	// Current tile index in gmem to read from
	int k_tile_next = 0;

	// Start async loads for all pipes but the last
	CUTE_UNROLL
	for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
		copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
		copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
		cp_async_fence();
		--k_tile_count;
		if (k_tile_count > 0) { ++k_tile_next; }
	}

	//
	// Define A/B partitioning and C accumulators
	//

	ThrMMA thr_mma = mma.get_slice(threadIdx.x);
	Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

	// Allocate registers for pipelining
	Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
	Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
	// Allocate the accumulators -- same size as the projected data
	Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

	CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
	CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
	CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

	// Clear the accumulators
	clear(tCrC);

	//
	// Copy Atom retiling
	//

	TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
	ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
	Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
	Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

	TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
	ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
	Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
	Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

#if 0
	if(thread0()) {
		print("  mA : "); print(  mA); print("\n");
		print("  gA : "); print(  gA); print("\n");
		print("  sA : "); print(  sA); print("\n");
		print("tAgA : "); print(tAgA); print("\n");
		print("tAsA : "); print(tAsA); print("\n");
	}
#endif

#if 0
	if(thread0()) {
		print("  mB : "); print(  mB); print("\n");
		print("  gB : "); print(  gB); print("\n");
		print("  sB : "); print(  sB); print("\n");
		print("tBgB : "); print(tBgB); print("\n");
		print("tBsB : "); print(tBsB); print("\n");
	}
#endif

#if 0
	if(thread0()) {
		print("  mC : "); print(  mC); print("\n");
		print("  gC : "); print(  gC); print("\n");
		print("tCgC : "); print(tCgC); print("\n");
		print("tCrA : "); print(tCrA); print("\n");
		print("tCrB : "); print(tCrB); print("\n");
		print("tCrC : "); print(tCrC); print("\n");

		print("tXsA : "); print(tXsA); print("\n");
		print("tXrA : "); print(tXrA); print("\n");
		print("tXsB : "); print(tXsB); print("\n");
		print("tXrB : "); print(tXrB); print("\n");
	}
#endif

#if 1

	// Current pipe index in smem to read from
	int smem_pipe_read  = 0;
	// Current pipe index in smem to write to
	int smem_pipe_write = K_PIPE_MAX-1;

	// Pipe slice
	Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
	Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

	// Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
	// Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

	// Size of the register pipeline
	auto K_BLOCK_MAX = size<2>(tCrA);

	// PREFETCH register pipeline
	if (K_BLOCK_MAX > 1) {
		// Wait until our first prefetched tile is loaded in
		cp_async_wait<K_PIPE_MAX-2>();
		__syncthreads();

		// Prefetch the first rmem from the first k-tile
		copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
		copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
	}

	//
	// PIPELINED MAIN LOOP
	// TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
	//           and explicit pipelines in shared memory.
	//   Data is read from global(k_tile_next) to shared(smem_pipe_write).
	//   Data is read from shared(smem_pipe_read) to registers(k_block_next).
	//   Data is computed on registers(b_block).
	//
	//   This allows all copies and compute to overlap:
	//     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
	//     Copy from smem->rmem can overlap with compute on rmem.
	//

	CUTE_NO_UNROLL
	while (k_tile_count > -(K_PIPE_MAX-1))
	{
		CUTE_UNROLL
		for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
		{
			if (k_block == K_BLOCK_MAX - 1)
			{
				// Slice the smem_pipe_read smem
				tXsA_p = tXsA(_,_,_,smem_pipe_read);
				tXsB_p = tXsB(_,_,_,smem_pipe_read);

				// Commit the smem for smem_pipe_read
				cp_async_wait<K_PIPE_MAX-2>();
				__syncthreads();
			}

			// Load A, B shmem->regs for k_block+1
			auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
			copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
			copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

			// Copy gmem to smem before computing gemm on each k-pipe
			if (k_block == 0)
			{
				copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
				copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
				cp_async_fence();

				// Advance the gmem tile
				--k_tile_count;
				if (k_tile_count > 0) { ++k_tile_next; }

				// Advance the smem pipe
				smem_pipe_write = smem_pipe_read;
				++smem_pipe_read;
				smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
				// smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
			}

			// Thread-level register gemm for k_block
			gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
		}
	}

#endif

	//
	// Epilogue
	//

	axpby(alpha, tCrC, beta, tCgC);
}

template <class ProblemShape, class CtaTiler,
					class TA, class AStride, class ASmemLayout, class TiledCopyA,
					class TB, class BStride, class BSmemLayout, class TiledCopyB,
					class TC, class CStride, class CSmemLayout, class TiledMma,
					class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
						TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
						TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
						TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
						Alpha alpha, Beta beta)
{
	using namespace cute;

	// Preconditions
	CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
	CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

	CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
	CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

	static_assert(is_static<ASmemLayout>::value);
	static_assert(is_static<BSmemLayout>::value);
	static_assert(is_static<CSmemLayout>::value);

	CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
	CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
	CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
	CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
	CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
	CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

	CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
	CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
	CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

	//
	// Full and Tiled Tensors
	//

	// Represent the full tensors
	Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
	Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
	Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

	// Get the appropriate blocks for this thread block
	auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
	Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
	Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
	Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

//   
//   __shared__ TA smemA[cosize_v<ASmemLayout>];
//   __shared__ TB smemB[cosize_v<BSmemLayout>];

	// Shared memory buffers
	extern __shared__ char shared_memory[];
	using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
	SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
	Tensor sA = make_tensor(make_smem_ptr(smem.A.data()), sA_layout);    // (BLK_M,BLK_K,PIPE)
	Tensor sB = make_tensor(make_smem_ptr(smem.B.data()), sB_layout);    // (BLK_N,BLK_K,PIPE)

	//
	// Partition the copying of A and B tiles across the threads
	//

	ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
	Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
	Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

	ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
	Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
	Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

	CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
	CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
	CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
	CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

	//
	// PREFETCH
	//

	auto K_PIPE_MAX = size<3>(tAsA);

	// Total count of tiles
	int k_tile_count = size<3>(tAgA);
	// Current tile index in gmem to read from
	int k_tile_next = 0;

	// Start async loads for all pipes but the last
	CUTE_UNROLL
	for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
		copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
		copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
		cp_async_fence();
		--k_tile_count;
		if (k_tile_count > 0) { ++k_tile_next; }
	}

	//
	// Define A/B partitioning and C accumulators
	//

	ThrMMA thr_mma = mma.get_slice(threadIdx.x);
	Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
	Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
	Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

	// Allocate registers for pipelining
	Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));                // (MMA,MMA_M,MMA_K)
	Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));                // (MMA,MMA_N,MMA_K)
	// Allocate the accumulators -- same size as the projected data
	Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

	auto identity_tensor_C{cute::make_identity_tensor(
		cute::make_shape(cute::size<0>(gC),
										 cute::size<1>(gC)))};

	auto thread_layout_C_identity_tensor_C{
		thr_mma.partition_C(identity_tensor_C)}; // (MMA, MMA_M, MMA_N)
	
	auto thread_layout_C_predicate_tensor_C{cute::make_tensor<bool>(
		cute::shape(thread_layout_C_identity_tensor_C))};

	CUTE_UNROLL
	for (auto i{0}; i < cute::size(thread_layout_C_predicate_tensor_C); ++i) {
		auto row_idx = cute::get<0>(thread_layout_C_identity_tensor_C(i)) + blockIdx.x * cute::size<0>(gC);
		auto col_idx = cute::get<1>(thread_layout_C_identity_tensor_C(i)) + blockIdx.y * cute::size<1>(gC);

		auto in_triangle = row_idx >= col_idx;
		auto in_bounds = row_idx < cute::size<0>(shape_MNK) && col_idx < cute::size<1>(shape_MNK);
		// Lower triangle
		thread_layout_C_predicate_tensor_C(i) = in_triangle && in_bounds;

		// Upper triangle
		// thread_layout_C_predicate_tensor_C(i) = row_idx <= col_idx;
					// cute::get<0>(thread_layout_C_identity_tensor_C(i)) +
					//         blockIdx.x * cute::size<0>(gC) <=
					// cute::get<1>(thread_layout_C_identity_tensor_C(i)) +
					//         blockIdx.y * cute::size<1>(gC);
	}

	// CUTE_UNROLL
	// for (auto i{0}; i < cute::size(thread_layout_C_predicate_tensor_C); ++i)
	// {
	//     thread_layout_C_predicate_tensor_C(i) = 
	//         cute::get<0>(thread_layout_C_identity_tensor_C(i)) +
	//                 blockIdx.x * cute::size<0>(gC) <
	//             (cute::size<0>(shape_MNK) - 2) &&
	//         cute::get<1>(thread_layout_C_identity_tensor_C(i)) +
	//                 blockIdx.y * cute::size<1>(gC) <
	//             (cute::size<1>(shape_MNK) - 2);
	// }

	CUTE_STATIC_ASSERT_V((  shape(tCrA) == take<0,3>(shape(tCsA))));     // (MMA,MMA_M,MMA_K)
	CUTE_STATIC_ASSERT_V((  shape(tCrB) == take<0,3>(shape(tCsB))));     // (MMA,MMA_N,MMA_K)
	CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
	CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));              // MMA_M
	CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));              // MMA_N
	CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));              // MMA_K

	// Clear the accumulators
	clear(tCrC);

#if 0
	if(thread0()) {
		print("  mA : "); print(  mA); print("\n");
		print("  gA : "); print(  gA); print("\n");
		print("  sA : "); print(  sA); print("\n");
		print("tAgA : "); print(tAgA); print("\n");
		print("tAsA : "); print(tAsA); print("\n");
	}
#endif

#if 0
	if(thread0()) {
		print("  mB : "); print(  mB); print("\n");
		print("  gB : "); print(  gB); print("\n");
		print("  sB : "); print(  sB); print("\n");
		print("tBgB : "); print(tBgB); print("\n");
		print("tBsB : "); print(tBsB); print("\n");
	}
#endif

#if 0
	if(thread0()) {
		print("  mC : "); print(  mC); print("\n");
		print("  gC : "); print(  gC); print("\n");
		print("tCsA : "); print(tCsA); print("\n");
		print("tCsB : "); print(tCsB); print("\n");
		print("tCgC : "); print(tCgC); print("\n");
		print("tCrA : "); print(tCrA); print("\n");
		print("tCrB : "); print(tCrB); print("\n");
		print("tCrC : "); print(tCrC); print("\n");
	}
#endif

#if 1

	// Current pipe index in smem to read from
	int smem_pipe_read  = 0;
	// Current pipe index in smem to write to
	int smem_pipe_write = K_PIPE_MAX-1;

	// Pipe slice
	Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
	Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

	// Size of the register pipeline
	auto K_BLOCK_MAX = size<2>(tCrA);

	// PREFETCH register pipeline
	if (K_BLOCK_MAX > 1) {
		// Wait until our first prefetched tile is loaded in
		cp_async_wait<K_PIPE_MAX-2>();
		__syncthreads();

		// Prefetch the first rmem from the first k-tile
		copy(tCsA_p(_,_,Int<0>{}), tCrA(_,_,Int<0>{}));
		copy(tCsB_p(_,_,Int<0>{}), tCrB(_,_,Int<0>{}));
	}

	//
	// PIPELINED MAIN LOOP
	// TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
	//           and explicit pipelines in shared memory.
	//   Data is read from global(k_tile_next) to shared(smem_pipe_write).
	//   Data is read from shared(smem_pipe_read) to registers(k_block_next).
	//   Data is computed on registers(b_block).
	//
	//   This allows all copies and compute to overlap:
	//     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
	//     Copy from smem->rmem can overlap with compute on rmem.
	//

	CUTE_NO_UNROLL
	while (k_tile_count > -(K_PIPE_MAX-1))
	{
		CUTE_UNROLL
		for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
		{
			if (k_block == K_BLOCK_MAX - 1)
			{
				// Slice the smem_pipe_read smem
				tCsA_p = tCsA(_,_,_,smem_pipe_read);
				tCsB_p = tCsB(_,_,_,smem_pipe_read);

				// Commit the smem for smem_pipe_read
				cp_async_wait<K_PIPE_MAX-2>();
				__syncthreads();
			}

			// Load A, B shmem->regs for k_block+1
			auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
			copy(tCsA_p(_,_,k_block_next), tCrA(_,_,k_block_next));
			copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));
			// Copy gmem to smem before computing gemm on each k-pipe
			if (k_block == 0)
			{
				copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
				copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
				cp_async_fence();

				// Advance the gmem tile
				--k_tile_count;
				if (k_tile_count > 0) { ++k_tile_next; }

				// Advance the smem pipe
				smem_pipe_write = smem_pipe_read;
				++smem_pipe_read;
				smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
			}
			// Thread-level register gemm for k_block
			gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
		}

	}

#endif

	//
	// Epilogue
	//

	// axpby(alpha, tCrC, beta, tCgC, thread_layout_C_predicate_tensor_C);
	axpby(alpha, tCrC, beta, tCgC);
}

template <
	class TA, class TB, class TC,
	class Alpha, class Beta
>
static
void
gemm_nt(
	int m, int n, int k,
	Alpha alpha,
	TA const* A, int ldA,
	TB const* B, int ldB,
	Beta beta,
	TC      * C, int ldC,
	cudaStream_t stream = 0
) {
	using namespace cute;

	// Define shapes (dynamic)
	auto M = int(m);
	auto N = int(n);
	auto K = int(k);
	auto prob_shape = make_shape(M, N, K);

	// Define NT strides (mixed)
	auto dA = make_stride(Int<1>{}, ldA);
	auto dB = make_stride(Int<1>{}, ldB);
	auto dC = make_stride(Int<1>{}, ldC);

	// Define CTA tile sizes (static)
	auto bM = Int<128>{};
	auto bN = Int<128>{};
	auto bK = Int< 32>{};
	auto cta_tiler = make_shape(bM, bN, bK);
	auto bP = Int<3>{};

	// Best with simple stride
	// auto swizzle_atom = composition(
	// 	Swizzle<2,3,2>{},
	// 	Layout<
	//     Shape<_32,_8>,
	//     Stride<_1,_32>
	//   >{}
	// );

	// Best with complex stride, no conflicts
	auto swizzle_atom = composition(
		Swizzle<3,2,4>{},
		Layout<
			Shape<_8, Shape<_4,_8>>,
			Stride<_1, Stride<_8,_32>>
		>{}
	);

	// Best with complex stride, no conflicts
	// auto swizzle_atom = composition(
	// 	Swizzle<5,0,5>{},
	// 	Layout<
	//     Shape<_32,_8>,
	//     Stride<_1,_32>
	//   >{}
	// );

	auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
	auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
	auto sC = make_layout(make_shape(bM, bN));

	TiledCopy copyA = make_tiled_copy(
		Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
		Layout<Shape<_16,_8>, Stride<_1,_16>>{},
		Layout<Shape< _4,_1>>{}
	);

	TiledCopy copyB = make_tiled_copy(
		Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
		Layout<Shape<_16, _8>, Stride<_1,_16>>{},
		Layout<Shape< _4,_1>>{}
	);

	TiledMMA mmaC = 
		make_tiled_mma(
			SM80_16x8x8_F32TF32TF32F32_TN{},
			Layout<Shape<_2,_2,_1>, Stride<_2, _1, _1>>{},
			Tile<_32,_32,_16>{}
		);

	const size_t size_mmaC = size(mmaC);
	const size_t size_copyA = size(copyA);

#if 0
	print(copyA);
	print(copyB);
	print(mmaC);
#endif

#if 0
	print_latex(copyA);
	print_latex(copyB);
	print_latex(mmaC);
#endif
	
	int smem_size = int(sizeof(SharedStorage<tfloat32_t, tfloat32_t, decltype(sA), decltype(sB)>));
	dim3 dimBlock(size(mmaC));
	dim3 dimGrid(
		size(ceil_div(M, bM)),
		size(ceil_div(N, bN))
	);
	// dim3 dimGrid(1,1);

	auto kernel_fptr = 
		gemm_device<
			decltype(prob_shape), decltype(cta_tiler),
			float, decltype(dA), decltype(sA), decltype(copyA),
			float, decltype(dB), decltype(sB), decltype(copyB),
			float, decltype(dC), decltype(sC), decltype(mmaC),
			decltype(alpha), decltype(beta)
		>;

	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributeMaxDynamicSharedMemorySize, 
		smem_size
	);
	
	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributePreferredSharedMemoryCarveout, 
		100
	);

	kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
		prob_shape, cta_tiler,
		A, dA, sA, copyA,
		B, dB, sB, copyB,
		C, dC, sC, mmaC,
		alpha, beta
	);
}

template <class Alpha, class Beta>
static
void
gemm_tn(int m, int n, int k,
				Alpha alpha,
				float const* A, int ldA,
				float const* B, int ldB,
				Beta beta,
				float      * C, int ldC,
				cudaStream_t stream = 0)
{
	using namespace cute;

	// Define shapes (dynamic)
	auto M = int(m);
	auto N = int(n);
	auto K = int(k);
	auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

	// Define TN strides (mixed)
	auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
	auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
	auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

	// Define CTA tile sizes (static)
	auto bM = Int<128>{};
	auto bN = Int<128>{};
	auto bK = Int<32>{};
	auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
	auto bP = Int<3>{};  // Pipeline

		auto swizzle_atom = composition(
				Swizzle<3,2,3>{},
				Layout<
						Shape <_8,_32>,
						Stride<_32, _1>
				>{}
		);

	// auto swizzle_atom = composition(
	//   Swizzle<@1@,@2@,@3@>{},
	//   Layout<Shape <_@4@,Shape <_@5@, _@6@>>,
	//           Stride<_@7@,Stride<_1,_@8@>>
	//   >{}
	// );

	TiledCopy copyA = make_tiled_copy(
		Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
		Layout<
			Shape<_16,_8>,
			Stride<_8,_1>
		>{},
		Layout<Shape< _1,_4>>{}
	);
	TiledCopy copyB = make_tiled_copy(
		Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
		Layout<
			Shape<_16,_8>,
			Stride<_8,_1>
		>{},
		Layout<Shape< _1,_4>>{}
	);
	TiledMMA mmaC = make_tiled_mma(
		SM80_16x8x8_F32TF32TF32F32_TN{},
		Layout<
			Shape<_2,_2>
		>{},
		Tile<_32,_32,_8>{}
	);

	Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t> s2r_atom_A;
	Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t> s2r_atom_B;

	// Copy_Atom<DefaultCopy, tfloat32_t> s2r_atom_A;
	// Copy_Atom<DefaultCopy, tfloat32_t> s2r_atom_B;

	auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
	auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
	auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx
		
	int smem_size = int(sizeof(SharedStorage<tfloat32_t, tfloat32_t, decltype(sA), decltype(sB)>));
	dim3 dimBlock(size(mmaC));
	dim3 dimGrid(
		size(ceil_div(M, bM)),
		size(ceil_div(N, bN))
	);
	#if 1
	auto kernel_fptr = 
				gemm_device_s2r<
						decltype(prob_shape), decltype(cta_tiler),
						float, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
						float, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
						float, decltype(dC), decltype(sC), decltype(mmaC),
						decltype(alpha), decltype(beta)
				>;

	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributeMaxDynamicSharedMemorySize, 
		smem_size
	);
	
	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributePreferredSharedMemoryCarveout, 
		100
	);

	kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
			prob_shape, cta_tiler,
			A, dA, sA, copyA, s2r_atom_A,
			B, dB, sB, copyB, s2r_atom_B,
			C, dC, sC, mmaC,
			alpha, beta
	);
	#endif

	#if 0
	auto kernel_fptr = 
				gemm_device<
						decltype(prob_shape), decltype(cta_tiler),
						float, decltype(dA), decltype(sA), decltype(copyA),
						float, decltype(dB), decltype(sB), decltype(copyB),
						float, decltype(dC), decltype(sC), decltype(mmaC),
						decltype(alpha), decltype(beta)
				>;

	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributeMaxDynamicSharedMemorySize, 
		smem_size
	);
	
	cudaFuncSetAttribute(
		kernel_fptr,
		cudaFuncAttributePreferredSharedMemoryCarveout, 
		100
	);
	
	kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
			prob_shape, cta_tiler,
			A, dA, sA, copyA,
			B, dB, sB, copyB,
			C, dC, sC, mmaC,
			alpha, beta
	);
	#endif
}

int 
main(
	int argc, 
	char** argv
) {
	int M = 128;
	if (argc >= 2)
		sscanf(argv[1], "%d", &M);

	int N = 128;
	if (argc >= 3)
		sscanf(argv[2], "%d", &N);

	int K = 32;
	if (argc >= 4)
		sscanf(argv[3], "%d", &K);

	char transA = 'N';
	if (argc >= 5)
		sscanf(argv[4], "%c", &transA);

	char transB = 'T';
	if (argc >= 6)
		sscanf(argv[5], "%c", &transB);

	using T = float;

	T alpha = 1.0f;
	T beta  = 0.0f;

	std::cout << "M = " << M << std::endl;
	std::cout << "N = " << N << std::endl;
	std::cout << "K = " << K << std::endl;
	std::cout << "C = A^" << transA << " B^" << transB << std::endl;

	thrust::host_vector<T> h_A(M*K);
	thrust::host_vector<T> h_B(N*K);
	thrust::host_vector<T> h_C_cublas(M*N);
	thrust::host_vector<T> h_C_cute(M*N);

	for (int j = 0; j < M*K; ++j) h_A[j] = static_cast<T>( 2*(rand() / double(RAND_MAX)) - 1 );
	for (int j = 0; j < N*K; ++j) h_B[j] = static_cast<T>( 2*(rand() / double(RAND_MAX)) - 1 );
	for (int j = 0; j < M*N; ++j) h_C_cute[j] = static_cast<T>(-1);
	for (int j = 0; j < M*N; ++j) h_C_cublas[j] = static_cast<T>(-1);

	thrust::device_vector<T> d_A = h_A;
	thrust::device_vector<T> d_B = h_B;
	thrust::device_vector<T> d_C_cublas = h_C_cublas;
	thrust::device_vector<T> d_C_cute = h_C_cute;

	int ldA = 0, ldB = 0, ldC = M;

	if (transA == 'N') {
		ldA = M;
	} else if (transA == 'T') {
		ldA = K;
	} else {
		assert(false);
	}

	if (transB == 'N') {
		ldB = K;
	} else if (transB == 'T') {
		ldB = N;
	} else {
		assert(false);
	}

	// assert(transA == 'T' && transB == 'N');

	gemm_tn(
			M, N, K,
			alpha,
			d_A.data().get(), ldA,
			d_B.data().get(), ldB,
			beta,
			d_C_cute.data().get(), ldC
	);
	
	CUTE_CHECK_LAST();
	thrust::host_vector<T> cute_result = d_C_cute;

	cublasHandle_t cublas_handle;
	cublasCheck( cublasCreate(&cublas_handle) );
	cublasCheck( cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH) );

	cublasCheck(
		cublasSgemm(
			cublas_handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			M, N, K, &alpha, 
			d_A.data().get(), ldA,
			d_B.data().get(), ldB,
			&beta, 
			d_C_cublas.data().get(), ldC
		)
	);

	thrust::host_vector<T> cublas_result = d_C_cublas;

	bool raise = false;

	for (int j = 0; j < M*N; ++j) {
		T cute_v = cute_result[j];
		T cublas_v = cublas_result[j];

		double diff = fabs((double)cute_v - (double)cublas_v);
		if (diff > 1e-3) {
				printf("(%d) difference: %f, %f\n", j, cute_v, cublas_v);
				raise = true;
			}
		}

		printf("cublas:\n");
		
		for (int row = 0; row < M && row < 4; row++) {
			for (int col = 0; col < N && col < 4; col++) {
				printf("%f, ", cublas_result[col * M + row]);
			}
			printf("\n");
		}

		printf("cute:\n");
		
		for (int row = 0; row < M && row < 4; row++) {
			for (int col = 0; col < N && col < 4; col++) {
				printf("%f, ", cute_result[col * M + row]);
			}
			printf("\n");
		}


		if (raise) {
			exit(EXIT_FAILURE);
		}
		
		exit(EXIT_SUCCESS);
}
