# Setting up dependencies
1. Ensure you have prerequisites required FOR installing LLVM, along with ninja build tools.
2. `git clone --recursive https://github.com/llvm/Polygeist.git` in your HOME directory.
3. `cd Polygeist`, and do `mkdir build && cd build`
4. Run the below:
```
cmake -G Ninja ../llvm-project/llvm \
   -DLLVM_ENABLE_PROJECTS="clang;mlir" \
   -DLLVM_EXTERNAL_PROJECTS="polygeist" \
   -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=../ \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_BUILD_TYPE=DEBUG \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_INSTALL_PREFIX=$HOME/llvm-mlir-pgeist
```

5. Run `ninja -jN`, REPLACE `N` with number of desired cores. 
6. Run the below checks. All checks must either pass, be unsupported, or expectedly fail.
```
ninja check-mlir
ninja check-polygeist-opt
ninja check-cgeist
```
7. `ninja install`
8. `pip install numpy` (or `apt install python3-numpy`)

# Compiling MLIR-Poly-Tiling
0. The above installation must be done in `$HOME` directory.
1. Ensure that you are using the `clang` that you built with `llvm-mlir-pgeist` by running `export PATH=$HOME/llvm-mlir-pgeist/bin:$PATH`
3. Run `opt-build.sh` to build the pass in `lib/Transform/Affine` and generate the `mlir-poly-tiling-opt` wrapper

# Sweeping all configs at once
To benchmark sweep over using 1st level through Nth level tiling with OUR strategy, AND one or more **1-level** loop tiling baselines, run: `./sweep_over_tiles.sh <benchmark_name> <T> "<C1,C2,...CN>" "<tsizes1>" "<tsizes2>" ... "<tsizesM>"`. ONLY modify the angled bracket arguments, and do not keep the angled brackets. 
- `benchmark_name` is the name of the benchmark. We used `matmul` in our paper.
- `T` is the number of trials to run per configuration.
- `C1...CN` represents cache sizes for L1, L2, L3, ... LN, in BYTES, comma-separated. Make sure to remove the angled brackets, and keep the double quotes. (e.g. `"3200,6400"` for 2 levels of caching).
- EACH `"<tsizesi>", where i = 1...M,` is a specific configuration of tile sizes, ordered from the outermost to innermost loop nest, that you want to manually test as a baseline. It must be comma-separated, and surrounded by the double quotes outside the angled brackets (e.g. `"20,30,40"` can be `"<tsizes1>"`, and it'll tile the outermost loop as 20, second loop nest as 30, innermost loop nest as 40). Make sure to remove the angled brackets.


**Example**: `./sweep_over_tiles.sh matmul 5 "32000,64000" "20,20,20" "40,40,40"`.

**Expected Output:** In stdout, you will see the timing results for using 1 level of caching, 2 levels, .. N levels with our method. You will also see single-level tiling profiling results for each of the manually inputted tile size baselines `"tsizes1" ... "tsizesM"`. In `experiments/<benchmark_name>`, you will see the tiling artifacts (MLIR, LLVM IR, final executable) only for the last baseline that was run. Please see the next two sections for running individual tiling experiments so the desired artifacts can be seen.

# Running our method only
To run only our method on K levels of caching (note this doesn't test 1-level, 2-level, ... K-level separately, it only tests K-level directly), run the following steps. ONLY MODIFY the angled bracket arguments, and remove the angled brackets while doing so. The format of `C1..CK` is each level's cache size, in bytes, and the double quotes must be kept while angled brackets are removed. `benchmark_name` is the benchmark used, for us it is `matmul`. Define `EXPERIMENT_DIR` as a desired `EXPERIMENT_DIR` you want to save artifacts to. `T` is number of trials desired.

1. `./compile_analyze.sh benchmarks/<benchmark_name>.c kernel <EXPERIMENT_DIR> "<C1,C2,...CK>" > /dev/null 2>&1`
2. `source <EXPERIMENT_DIR>/tile_size_env_vars.sh`
3. `./tile_postproc_ours.sh <EXPERIMENT_DIR>`
4. `./profile.sh <EXPERIMENT_DIR> <T>`

Timings for just the K-level caching will be outputted, and the desired artifacts (namely, a `tiled_file.mlir` with the tiling applied) will be under EXPERIMENT_DIR.

# Running on baselines

To run only a baseline of fixed tile sizes pre-determined, run the following steps. ONLY MODIFY the angled bracket arguments, and remove the angled brackets while doing so. `benchmark_name` is the benchmark used, for us it is `matmul`. Define `EXPERIMENT_DIR` as a desired `EXPERIMENT_DIR` you want to save artifacts to. `T` is number of trials desired.

1. `./compile_analyze.sh benchmarks/<benchmark_name>.c kernel <EXPERIMENT_DIR> "32000" > /dev/null 2>&1`
3. `./tile_postproc_baseline.sh <EXPERIMENT_DIR> "<outermostLoopNestTileSize, nextLoopNestTileSize, ..., innerMostLoopNestTileSize>" > /dev/null 2>&1`. Keep the double quotes around the tile size list, remove the angled brackets.
4. `./profile.sh <EXPERIMENT_DIR> <T>`

Timings for just the baseline will be outputted, and the desired artifacts (namely, a `tiled_file.mlir` with the tiling applied) will be under EXPERIMENT_DIR.
