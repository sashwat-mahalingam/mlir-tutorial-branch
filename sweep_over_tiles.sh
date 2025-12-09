
# $1: benchmark to run
# $2: num trials
# $3: cache sizes
# everything afterwards is a set of tile sizes to test out, e.g. "3,3,3" "4,4,4". These will be checked against the optimal tiling strategy.
#Tile sizes must be comma separated and wrapped in quotes.

BENCHMARK_C_FILE="benchmarks/$1.c"
BENCHMARK_FUNCTION="kernel"
BENCHMARK_EXP_DIR="experiments/$1"
NUM_TRIALS=$2
CACHE_SIZES=$3
DISABLE_OPTIMAL_TILING=${4:-0}

./compile_analyze.sh  ${BENCHMARK_C_FILE} ${BENCHMARK_FUNCTION} ${BENCHMARK_EXP_DIR} $CACHE_SIZES > /dev/null 2>&1;

source ${BENCHMARK_EXP_DIR}/tile_size_env_vars.sh
numLevels=$NUM_TILE_LEVELS

echo "Optimal tiling strategy, with $numLevels levels";
tail +2 ${BENCHMARK_EXP_DIR}/tile_size_env_vars.sh;

if [ $DISABLE_OPTIMAL_TILING -eq 0 ]; then
    for lvls in $(seq 1 $numLevels);
    do    
        echo "--------------------------------";
        echo "Num levels: $lvls";
        echo "   ";
        echo "   ";
        export NUM_TILE_LEVELS=$lvls
        ./tile_postproc_ours.sh ${BENCHMARK_EXP_DIR}
        ./profile.sh ${BENCHMARK_EXP_DIR} $NUM_TRIALS
        echo "     ";
        echo "--------------------------------";
    done
fi

for tile_size in ${@:5};
do
    echo "--------------------------------";
    echo "Tile size: $tile_size";
    echo "    ";
    ./tile_postproc_baseline.sh ${BENCHMARK_EXP_DIR} $tile_size > /dev/null 2>&1;
    ./profile.sh ${BENCHMARK_EXP_DIR} $NUM_TRIALS
    echo "    ";
    echo "--------------------------------"; 
done
