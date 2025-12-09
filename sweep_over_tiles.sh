
# Benchmark C file: benchmark to run
# benchmark function: function to analyze for tiling
# benchmark experiment directory: directory to store the experiment
# number of trials: number of trials to run for each tile size
# everything afterwards is a set of tile sizes to test out, e.g. "3,3,3" "4,4,4". 
#Tile sizes must be comma separated and wrapped in quotes.

BENCHMARK_C_FILE=$1
BENCHMARK_FUNCTION=$2
BENCHMARK_EXP_DIR=$3
NUM_TRIALS=$4

for tile_size in ${@:5};
do
    echo "--------------------------------";
    echo "Tile size: $tile_size";
    echo "    ";
    ./compile_analyze.sh  ${BENCHMARK_C_FILE} ${BENCHMARK_FUNCTION} ${BENCHMARK_EXP_DIR} > /dev/null 2>&1;
    ./tile_postproc.sh ${BENCHMARK_EXP_DIR} $tile_size > /dev/null 2>&1;
    ./profile.sh ${BENCHMARK_EXP_DIR} $NUM_TRIALS
    echo "    ";
    echo "--------------------------------"; 
done