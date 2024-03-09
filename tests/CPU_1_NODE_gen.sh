#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                      # number of tasks
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --cpus-per-task=128                  # number of cores (OpenMP thread) per task
#SBATCH --time=01:00:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --output=results/CPU_1_NODE_gen.out

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests/"
TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/test_CPU_MPI_OMP.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"

NODES=1
THREADS=256

echo "CPU TEST MPI+OMP GEN (nodes: $NODES - threads per node: $THREADS)"
echo "first-touch policy"

echo "generation"
srun -n $NODES -c $THREADS $TEST -s 80000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 90000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 100000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 110000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 120000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 140000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 160000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 180000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 200000 -o $OUTPUT -i 15 2>>errors.txt


TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build_no_fp/test/test_CPU_MPI_OMP.out"
echo "CPU TEST MPI+OMP GEN(nodes: $NODES - threads per node: $THREADS)"
echo "Without first-touch policy"
srun -n $NODES -c $THREADS $TEST -s 80000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 90000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 100000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 110000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 120000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -s 140000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 160000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 180000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES -c $THREADS $TEST -s 200000 -o $OUTPUT -i 15 2>>errors.txt