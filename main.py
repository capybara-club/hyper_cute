import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import itertools
import tempfile
from typing import List, Callable, Tuple

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

class ParameterCombinations:
    def __init__(self):
        # Define hyperparameter values
        self.swizzle_0s = swizzle_0s
        self.swizzle_1s = swizzle_1s
        self.swizzle_2s = swizzle_2s
        self.shape_Xs = shape_Xs
        self.shape_Ys = shape_Ys
        self.shape_Zs = shape_Zs
        self.stride_Xs = stride_Xs
        self.stride_Zs = stride_Zs
        self.tile_ks = tile_ks
        
        # Store all parameter names and their values
        self.param_dict = {
            'swizzle_0': self.swizzle_0s,
            'swizzle_1': self.swizzle_1s,
            'swizzle_2': self.swizzle_2s,
            'shape_X': self.shape_Xs,
            'shape_Y': self.shape_Ys,
            'shape_Z': self.shape_Zs,
            'stride_X': self.stride_Xs,
            'stride_Z': self.stride_Zs,
            'tile_k': self.tile_ks
        }
        
        # List to store filter functions
        self.filters: List[Callable[[dict], bool]] = []

    def add_filter(self, filter_func: Callable[[dict], bool]) -> None:
        """Add a filter function to the list of filters"""
        self.filters.append(filter_func)

    def get_filtered_combinations(self) -> List[Tuple]:
        """Generate all combinations and apply filters"""
        # Generate all possible combinations
        all_combinations = list(itertools.product(*self.param_dict.values()))
        
        # Convert to list of dicts for easier filtering
        param_names = list(self.param_dict.keys())
        dict_combinations = [
            dict(zip(param_names, comb)) 
            for comb in all_combinations
        ]
        
        # Apply all filters
        filtered_combinations = dict_combinations
        for filter_func in self.filters:
            filtered_combinations = [
                comb for comb in filtered_combinations 
                if filter_func(comb)
            ]
        
        # Convert back to tuples for ProcessPoolExecutor compatibility
        return [tuple(comb.values()) for comb in filtered_combinations]

param_combs = ParameterCombinations()
param_combs.add_filter(lambda x: x['swizzle_0'] >= x['swizzle_2'])

combinations = param_combs.get_filtered_combinations()

# Generate all combinations
# combinations = list(itertools.product(swizzle_0s, swizzle_1s, swizzle_2s, shape_Xs, shape_Ys, shape_Zs, stride_Xs, stride_Zs, tile_ks))
# print(combinations[:8])
print(len(combinations))
exit()

# Process combinations in parallel
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_set, i, comb) for i, comb in enumerate(combinations)]
    for future in futures:
        print(future.result())