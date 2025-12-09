export PATH=$HOME/Polygeist/build/bin:$PATH

BENCHMARK_FILE=$1
BENCHMARK_FUNCTION=$2
BENCHMARK_OUTPUT_DIR=$3
CACHE_SIZES=$4

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

OUTPUT_FILE=$BENCHMARK_OUTPUT_DIR/pre_tile_analysis_func.txt ./build-ninja/tools/mlir-poly-tiling-opt $BENCHMARK_OUTPUT_DIR/raised_func.mlir --pre-tile-analysis;
echo "done pre-tile analysis";
python3 calc_tile_sizes.py --pre_tile_analysis_file $BENCHMARK_OUTPUT_DIR/pre_tile_analysis_func.txt --cache_sizes $CACHE_SIZES