export PATH=$HOME/Polygeist/build/bin:$PATH

BENCHMARK_EXP_DIR=$1
TILE_SIZES=$2

mlir-opt $BENCHMARK_EXP_DIR/raised_file.mlir \
    --affine-loop-tile="tile-sizes=${TILE_SIZES}" \
    --canonicalize \
    --cse \
    -o $BENCHMARK_EXP_DIR/tiled_file.mlir
    
polygeist-opt $BENCHMARK_EXP_DIR/tiled_file.mlir --lower-affine | mlir-opt --convert-scf-to-cf | mlir-opt --convert-to-llvm | mlir-opt \
    --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o $BENCHMARK_EXP_DIR/tiled_ir_file.ll

clang -o $BENCHMARK_EXP_DIR/tiled_exec $BENCHMARK_EXP_DIR/tiled_ir_file.ll


