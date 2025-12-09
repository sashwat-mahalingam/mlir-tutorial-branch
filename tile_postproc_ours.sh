export PATH=$HOME/Polygeist/build/bin:$PATH

BENCHMARK_EXP_DIR=$1

source ${BENCHMARK_EXP_DIR}/tile_size_env_vars.sh

./build-ninja/tools/mlir-poly-tiling-opt $BENCHMARK_EXP_DIR/raised_file.mlir \
    --tile-size-selection \
    --canonicalize \
    --cse \
    -o $BENCHMARK_EXP_DIR/tiled_file.mlir
    
polygeist-opt $BENCHMARK_EXP_DIR/tiled_file.mlir --lower-affine | mlir-opt --convert-scf-to-cf | mlir-opt --convert-to-llvm | mlir-opt \
    --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o $BENCHMARK_EXP_DIR/tiled_ir_file.ll

clang -o $BENCHMARK_EXP_DIR/tiled_exec $BENCHMARK_EXP_DIR/tiled_ir_file.ll


