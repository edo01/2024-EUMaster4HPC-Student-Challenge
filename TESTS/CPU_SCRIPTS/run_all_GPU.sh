#!/bin/bash -l
#SBATCH --nodes=16                          # number of nodes
#SBATCH --ntasks=64                      # number of tasks
#SBATCH --cpus-per-task=4                  # number of cores (OpenMP thread) per task
#SBATCH --time=00:10:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests/"
TEST="/home/users/u101379/tests/test_CG_MultiGPUS_CUDA_MPI.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"


#srun -n 64 $TEST -s 560000 -o $OUTPUT -i 3000

#echo "strong scalability test 50000"

#srun -n 12 $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT
#srun -n 16 $TEST -A $MATRIXDIR/matrix50000.bin -b $MATRIXDIR/rhs50000.bin -o $OUTPUT

#echo "strong scalability test 60000"

#srun -n 12 $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT  
#srun -n 16 $TEST -A $MATRIXDIR/matrix60000.bin -b $MATRIXDIR/rhs60000.bin -o $OUTPUT

#echo "strong scalability test 70000"
#srun -n 12 $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT
#srun -n 16 $TEST -A $MATRIXDIR/matrix70000.bin -b $MATRIXDIR/rhs70000.bin -o $OUTPUT

#weak scalability test
#echo "weak scalability test"

#srun -n 1 $TEST -A  $MATRIXDIR/matrix10000.bin  -b $MATRIXDIR/rhs10000.bin -o $OUTPUT
#srun -n 4  $TEST -A  $MATRIXDIR/matrix20000.bin  -b $MATRIXDIR/rhs20000.bin -o $OUTPUT
#srun -n 16 $TEST -A  $MATRIXDIR/matrix40000.bin  -b $MATRIXDIR/rhs40000.bin -o $OUTPUT
#srun -n 48 $TEST -A  $MATRIXDIR/matrix70000.bin  -b $MATRIXDIR/rhs70000.bin -o $OUTPUT
srun -n 64 $TEST   -s 80000  -o $OUTPUT -i 1000
#srun -n 32 $TEST -s 160000  -o $OUTPUT -i 30
#srun -n 64 $TEST -s 320000 -o $OUTPUT -i 30
