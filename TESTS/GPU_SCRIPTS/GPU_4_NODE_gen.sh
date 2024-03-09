#!/bin/bash -l
#SBATCH --nodes=4                          # number of nodes
#SBATCH --ntasks=16                      # number of tasks
#SBATCH --ntasks-per-node=4               # number of tasks per node
#SBATCH --time=00:05:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --output=results/GPU_4_NODE_gen.out

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests/"
TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/test_CG_MultiGPUS_CUDA_MPI.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"

NODES=16

#echo "GPU TEST CUDA GEN (nodes: $NODES)"
#echo "CUDA+MPI"

#srun -n $NODES $TEST -s 80000 -o $OUTPUT -i 15 2>>errors.txt 
#srun -n $NODES $TEST -s 90000 -o $OUTPUT -i 15 2>>errors.txt 
#srun -n $NODES $TEST -s 100000 -o $OUTPUT -i 15 2>>errors.txt 
#srun -n $NODES $TEST -s 110000 -o $OUTPUT -i 15 2>>errors.txt 
#srun -n $NODES $TEST -s 120000 -o $OUTPUT -i 15 2>>errors.txt 
#srun -n $NODES $TEST -s 140000 -o $OUTPUT -i 15 2>>errors.txt
#srun -n $NODES $TEST -s 160000 -o $OUTPUT -i 15 2>>errors.txt
#srun -n $NODES $TEST -s 180000 -o $OUTPUT -i 15 2>>errors.txt
#srun -n $NODES $TEST -s 200000 -o $OUTPUT -i 15 2>>errors.txt


TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/test_CG_MultiGPUS_CUDA_NCCL.out"
echo "CPU TEST CUDA GEN (nodes: $NODES)"
echo "NCCL+MPI"
srun -n $NODES $TEST -s 80000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES $TEST -s 90000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES $TEST -s 100000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES $TEST -s 110000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES $TEST -s 120000 -o $OUTPUT -i 15 2>>errors.txt 
srun -n $NODES $TEST -s 140000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES $TEST -s 160000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES $TEST -s 180000 -o $OUTPUT -i 15 2>>errors.txt
srun -n $NODES $TEST -s 200000 -o $OUTPUT -i 15 2>>errors.txt