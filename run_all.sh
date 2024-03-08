#!/bin/bash -l
#SBATCH --nodes=20                          # number of nodes
#SBATCH --ntasks=20                      # number of tasks
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --cpus-per-task=128                  # number of cores (OpenMP thread) per task
#SBATCH --time=00:15:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default

module load CUDA UCX OpenMPI CMake NCCL

MATRIXDIR="/project/home/p200301/tests/"
TEST="/home/users/u101379/tests/test_CPU_MPI_OMP.out"
OUTPUT="/home/users/u101379/2024-EUMaster4HPC-Student-Challenge/challenge/io/sol.bin"

echo "first-touch policy"

echo "strong scalability test 20000"

for i in 2 4 6 8 10 12 14 16 18 20; do
    srun -n $i -c 256 $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt 
done

#scalability test 
echo "strong scalability test 30000"

for i in 2 4 6 8 10 12 14 16 18 20; do
    srun -n $i -c 256 $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt
done

# scalability thread
echo "strong scalability thread test 10000"
for i in 1 2 4 8 16 32 64 128 256; do
    srun -n 1 -c $i $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt
done

#weak scalability test
#echo "weak scalability test"

#srun -n 1 -c 256 $TEST -A  $MATRIXDIR/matrix10000.bin  -b $MATRIXDIR/rhs10000.bin -o $OUTPUT
#srun -n 2 -c 256 $TEST -A  $MATRIXDIR/matrix20000.bin  -b $MATRIXDIR/rhs20000.bin -o $OUTPUT
#srun -n 4 -c 256 $TEST -A  $MATRIXDIR/matrix40000.bin  -b $MATRIXDIR/rhs40000.bin -o $OUTPUT
#srun -n 8 -c 256 $TEST -A  $MATRIXDIR/matrix80000.bin  -b $MATRIXDIR/rhs80000.bin -o $OUTPUT
#srun -n 16 -c 256 $TEST -A  $MATRIXDIR/matrix160000.bin  -b $MATRIXDIR/rhs160000.bin -o $OUTPUT
#srun -n 32 -c 256 $TEST -A  $MATRIXDIR/matrix320000.bin  -b $MATRIXDIR/rhs320000.bin -o $OUTPUT


TEST="/home/users/u101379/tests/no_ft/test_CPU_MPI_OMP.out"

echo 
echo
echo "NO first-touch policy"

echo "strong scalability test 20000"

for i in 2 4 6 8 10 12 14 16 18 20; do
    srun -n $i -c 256 $TEST -A $MATRIXDIR/matrix20000.bin -b $MATRIXDIR/rhs20000.bin -o $OUTPUT  2>>errors.txt
done

#scalability test 
echo "strong scalability test 30000"

for i in 2 4 6 8 10 12 14 16 18 20; do
    srun -n $i -c 256 $TEST -A $MATRIXDIR/matrix30000.bin -b $MATRIXDIR/rhs30000.bin -o $OUTPUT  2>>errors.txt
done

# scalability thread
echo "strong scalability thread test 10000"
for i in 1 2 4 8 16 32 64 128 256; do
    srun -n 1 -c $i $TEST -A $MATRIXDIR/matrix10000.bin -b $MATRIXDIR/rhs10000.bin -o $OUTPUT  2>>errors.txt
done


#weak scalability test
#echo "weak scalability test"

#srun -n 1 -c 256 $TEST -A  $MATRIXDIR/matrix10000.bin  -b $MATRIXDIR/rhs10000.bin -o $OUTPUT
#srun -n 2 -c 256 $TEST -A  $MATRIXDIR/matrix20000.bin  -b $MATRIXDIR/rhs20000.bin -o $OUTPUT
#srun -n 4 -c 256 $TEST -A  $MATRIXDIR/matrix40000.bin  -b $MATRIXDIR/rhs40000.bin -o $OUTPUT
#srun -n 8 -c 256 $TEST -A  $MATRIXDIR/matrix80000.bin  -b $MATRIXDIR/rhs80000.bin -o $OUTPUT
#srun -n 16 -c 256 $TEST -A  $MATRIXDIR/matrix160000.bin  -b $MATRIXDIR/rhs160000.bin -o $OUTPUT
#srun -n 32 -c 256 $TEST -A  $MATRIXDIR/matrix320000.bin  -b $MATRIXDIR/rhs320000.bin -o $OUTPUT