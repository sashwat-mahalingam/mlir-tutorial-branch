export PATH=$HOME/Polygeist/build/bin:$PATH

BENCHMARK_FILE=$1
BENCHMARK_FUNCTION=$2
BENCHMARK_OUTPUT_DIR=$3

if [ -d $BENCHMARK_OUTPUT_DIR ]; then
    rm -rf $BENCHMARK_OUTPUT_DIR;
fi

# recursively create the directory
mkdir -p $BENCHMARK_OUTPUT_DIR;

for option in 1 2; do
    if [ $option -eq 1 ]; then
        setting="_func";
        function=$BENCHMARK_FUNCTION;
    else
        setting="_file";
        function="*";
    fi

    cgeist $BENCHMARK_FILE -I $HOME/llvm-mlir-pgeist/lib/clang/18/include \
    --function=$function -S | polygeist-opt --raise-scf-to-affine --polygeist-mem2reg -o $BENCHMARK_OUTPUT_DIR/unraised${setting}.mlir;

    ./build-ninja/tools/mlir-poly-tiling-opt $BENCHMARK_OUTPUT_DIR/unraised${setting}.mlir --raise-to-affine -o $BENCHMARK_OUTPUT_DIR/raised${setting}.mlir

done

./build-ninja/tools/mlir-poly-tiling-opt $BENCHMARK_OUTPUT_DIR/raised_func.mlir --pre-tile-analysis -o $BENCHMARK_OUTPUT_DIR/pre_tile_analysis_func.mlir