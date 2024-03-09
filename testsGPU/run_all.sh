#!/bin/bash 

# Merge all files ending with .res
cat results/*.res > test$1.txt

# Remove blank lines from the merged file
sed -i '/^\s*$/d' test$1.txt

# Merge all files ending with .res
cat results/*.out > test_gen$1.txt

# Remove blank lines from the merged file
sed -i '/^\s*$/d' test_gen$1.txt

# This script is used to run all the tests in the tests folder
#sbatch GPU_1_NODE.sh
#sbatch GPU_2_NODE.sh
#sbatch GPU_3_NODE.sh
#sbatch GPU_4_NODE.sh
#sbatch GPU_8_NODE.sh
#sbatch GPU_16_NODE.sh

#sbatch GPU_1_NODE_gen.sh
#sbatch GPU_2_NODE_gen.sh
#sbatch GPU_3_NODE_gen.sh
#sbatch GPU_4_NODE_gen.sh
#sbatch GPU_8_NODE_gen.sh
#sbatch GPU_16_NODE_gen.sh