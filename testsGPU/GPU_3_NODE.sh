#!/bin/bash -l
#SBATCH --nodes=3                         # number of nodes
#SBATCH --ntasks=12                      # number of tasks
#SBATCH --ntasks-per-node=4               # number of tasks per node
#SBATCH --time=00:10:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --output=results/GPU_3_NODE.res

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests"
TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/test_CG_MultiGPUS_CUDA_MPI.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"

NODES=12

echo "GPU TEST CUDA (nodes: $NODES)"
echo "CUDA+MPI"
srun -n $NODES $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix40000.bin -b $MATRIXDIR/rhs40000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT  2>>errors.txt 


TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build_no_fp/test/test_CG_MultiGPUS_CUDA_NCCL.out"
echo "CPU TEST CUDA (nodes: $NODES)"
echo "NCCL+MPI"
srun -n $NODES $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix40000.bin -b $MATRIXDIR/rhs40000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT  2>>errors.txt 