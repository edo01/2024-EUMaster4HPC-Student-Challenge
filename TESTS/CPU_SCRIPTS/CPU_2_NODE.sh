#!/bin/bash -l
#SBATCH --nodes=2                          # number of nodes
#SBATCH --ntasks=2                      # number of tasks
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --cpus-per-task=128                  # number of cores (OpenMP thread) per task
#SBATCH --time=01:00:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --output=results/CPU_2_NODE.res

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests/"
TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build/test/test_CPU_MPI_OMP.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"

NODES=2
THREADS=256

echo "CPU TEST MPI+OMP (nodes: $NODES - threads per node: $THREADS)"
echo "first-touch policy"
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix40000.bin -b $MATRIXDIR/rhs40000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT  2>>errors.txt 
#srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT  2>>errors.txt 
#srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT  2>>errors.txt 


TEST="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/main/build_no_fp/test/test_CPU_MPI_OMP.out"
echo "CPU TEST MPI+OMP (nodes: $NODES - threads per node: $THREADS)"
echo "Without first-touch policy"
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix40000.bin -b $MATRIXDIR/rhs40000.bin -o $OUTPUT  2>>errors.txt 
srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT  2>>errors.txt 
#srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT  2>>errors.txt 
#srun -n $NODES -c $THREADS $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT  2>>errors.txt 