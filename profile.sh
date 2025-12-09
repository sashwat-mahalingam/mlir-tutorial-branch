BENCHMARK_EXP_DIR=$1
NUM_TRIALS=$2

overall_time=0;
for i in $(seq 1 $NUM_TRIALS);
do
    curr_time=$(./${BENCHMARK_EXP_DIR}/tiled_exec);
    curr_time=$(echo $curr_time | bc -l);
    echo $curr_time;
    overall_time=$(echo "$overall_time + $curr_time" | bc -l);
done
echo "     ";
echo "Average time: $(echo "scale=2; $overall_time / $NUM_TRIALS" | bc -l) seconds";
