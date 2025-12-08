# Set up instructions
0. The following must be in your home directory: $HOME/Polygeist (built). Installed binaries for Polygeist under $HOME/llvm-mlir-pgeist.
1. Ensure that you are using the clang that you built with llvm-mlir-pgeist by running "export PATH=$HOME/llvm-mlir-pgeist/bin:$PATH"
3. Run opt-build.sh to build the pass in lib/Transform/Affine and generate the mlir-poly-tiling-opt wrapper
4. Run run.sh (Refer to PreTileAnalysis.h and PreTileAnalysis.cpp)