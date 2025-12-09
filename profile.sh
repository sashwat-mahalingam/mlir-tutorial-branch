BENCHMARK_EXP_DIR=$1

overall_time=0;
for i in 1 2 3 4 5 6 7 8 9 10;
do
    curr_time=$(./${BENCHMARK_EXP_DIR}/tiled_exec);
    curr_time=$(echo $curr_time | bc -l);
    echo $curr_time;
    overall_time=$(echo "$overall_time + $curr_time" | bc -l);
done
echo "Average time: $(echo "scale=2; $overall_time / 10" | bc -l) seconds";
