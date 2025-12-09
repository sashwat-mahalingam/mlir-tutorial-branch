BENCHMARK_C_FILE=$1
BENCHMARK_EXP_DIR=$2

for tile_size in ${@:3};
do
    echo "--------------------------------";
    echo "Tile size: $tile_size";
    echo "    ";
    ./compile_analyze.sh  ${BENCHMARK_C_FILE} matmul ${BENCHMARK_EXP_DIR} > /dev/null 2>&1;
    ./tile_postproc.sh ${BENCHMARK_EXP_DIR} $tile_size > /dev/null 2>&1;
    ./profile.sh ${BENCHMARK_EXP_DIR}
    echo "    ";
    echo "--------------------------------"; 
done