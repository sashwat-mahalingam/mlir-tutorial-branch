#!/bin/sh
FIRST_TIME_BUILD=$1

if [ "$FIRST_TIME_BUILD" = "--first-time" ]; then
    first_time_build=true
    build_deps_flag="ON"
else
    first_time_build=false
    build_deps_flag="OFF"
fi

BUILD_SYSTEM="Ninja"
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

BUILD_SYSTEM="Ninja"

if [ "$first_time_build" = true ]; then
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
else
    cd $BUILD_DIR
fi

LLVM_BUILD_DIR=${HOME}/Polygeist/build/
cmake -G $BUILD_SYSTEM .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS=${build_deps_flag} \
    -DBUILD_SHARED_LIBS="OFF" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_INCLUDE_DIRS=${HOME}/llvm-mlir-pgeist/include/llvm \
    -DMLIR_INCLUDE_DIRS=${HOME}/llvm-mlir-pgeist/include/mlir && cmake --build . --target mlir-poly-tiling-opt --parallel ${1:-2}