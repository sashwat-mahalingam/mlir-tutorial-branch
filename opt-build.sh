#!/bin/sh
BUILD_SYSTEM="Ninja"
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

LLVM_BUILD_DIR=${HOME}/Polygeist/build/
cmake -G $BUILD_SYSTEM .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS="ON" \
    -DBUILD_SHARED_LIBS="OFF" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_INCLUDE_DIRS=${HOME}/llvm-mlir-pgeist/include/llvm \
    -DMLIR_INCLUDE_DIRS=${HOME}/llvm-mlir-pgeist/include/mlir && cmake --build . --target mlir-poly-tiling-opt --parallel ${1:-2}