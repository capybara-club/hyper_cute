import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import itertools
import tempfile

def process_set(set_id, numbers):
    """
    Process a single combination by updating the C++ source code, compiling, and running it.

    Args:
        set_id (int): Unique identifier for the directory.
        numbers (tuple): Tuple of values (x, y) to use in the source code.

    Returns:
        str: Result message for the combination.
    """
    # Create a unique directory
    with tempfile.TemporaryDirectory() as dir_name:
        # Copy the template to the temporary directory
        source_file = os.path.join(dir_name, "main.cu")
        shutil.copy("template.cu", source_file)
    # dir_name = f"set{set_id}"
    # os.makedirs(dir_name, exist_ok=True)

        # Replace placeholders with values
        with open(source_file, "r") as f:
            content = f.read()
        for i, num in enumerate(numbers, start=1):
            content = content.replace(f"@{i}@", str(num))
        with open(source_file, "w") as f:
            f.write(content)

        # Compile the code
        compile_result = subprocess.run(
            ["nvcc",
            "-arch=native",
            "-I/home/cdurham/hyper_cute/cutlass/include",
            "-I/home/cdurham/hyper_cute/cutlass/tools/util/include",
            "--compiler-options=-fno-omit-frame-pointer",
            "--expt-relaxed-constexpr",
            "-lcublas",
            "main.cu", 
            "-o",
            "main"
            ],
            cwd=dir_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if compile_result.returncode != 0:
            return f"combination {numbers}: compilation failed"

        # Run the compiled program
        run_result = subprocess.run(
            ["./main"],
            cwd=dir_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if run_result.returncode == 0:
            return f"combination {numbers}: success"
        else:
            return f"combination {numbers}: program failed with code {run_result.returncode}"

# Define hyperparameter values
swizzle_0s = [0,1,2,3,4,5,6]
swizzle_1s = [0,1,2,3,4,5,6]
swizzle_2s = [0,1,2,3,4,5,6]
shape_Xs = [4,8]
shape_Ys = [4,8]
shape_Zs = [2,4,8]
stride_Xs = [4,8,16,32]
stride_Zs = [4,8,16,32,64]
tile_ks = [8,16,32]

swizzle_0s = [0]
swizzle_1s = [0]
swizzle_2s = [0]
shape_Xs = [4]
shape_Ys = [4]
shape_Zs = [2]
stride_Xs = [4]
stride_Zs = [4]
tile_ks = [8]

# Generate all combinations
combinations = list(itertools.product(swizzle_0s, swizzle_1s, swizzle_2s, shape_Xs, shape_Ys, shape_Zs, stride_Xs, stride_Zs, tile_ks))

print(len(combinations))
# exit()

# Process combinations in parallel
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_set, i, comb) for i, comb in enumerate(combinations)]
    for future in futures:
        print(future.result())