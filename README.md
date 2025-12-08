# Setting up dependencies
1. Ensure you have prerequisites required FOR installing LLVM, along with ninja build tools.
2. `git clone --recursive https://github.com/llvm/Polygeist.git` in HOME directory.
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


# Compiling/running MLIR-Poly-Tiling
0. The above installation must be done in `$HOME` directory.
1. Ensure that you are using the `clang` that you built with `llvm-mlir-pgeist` by running `export PATH=$HOME/llvm-mlir-pgeist/bin:$PATH`
3. Run `opt-build.sh` to build the pass in `lib/Transform/Affine` and generate the `mlir-poly-tiling-opt` wrapper
4. Run `run.sh` (Refer to passes in `lib/Transform/Affine`)