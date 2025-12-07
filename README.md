# Set up instructions
1. Ensure that you are using the clang that you built with llvm-mlir-polygeist by running "export PATH=/home/userName/llvm-mlir-pgeist/bin:$PATH" REPLACE YOUR USERNAME
2. Replace userName with your path in opt-build.sh and run.sh
3. Run opt-build.sh to build the pass in lib/Transform/Affine and generate the tutorial-opt wrapper
4. Run run.sh
5. Refer to AffineFullUnroll.cpp (Don't mind the name, the pass actually has nothing to do with Loop Unrolling)