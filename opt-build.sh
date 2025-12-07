#!/bin/sh

BUILD_SYSTEM="Ninja"
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

BUILD_SYSTEM="Ninja"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
pushd $BUILD_DIR
cd $BUILD_DIR

LLVM_BUILD_DIR=/home/userName/Polygeist/build/
cmake -G $BUILD_SYSTEM .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS="ON" \
    -DBUILD_SHARED_LIBS="OFF" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/home/userName/mlir-custom-built \
    -DLLVM_INCLUDE_DIRS=/home/userName/llvm-mlir-pgeist/include/llvm \
    -DMLIR_INCLUDE_DIRS=/home/userName/llvm-mlir-pgeist/include/mlir && cmake --build . --target tutorial-opt