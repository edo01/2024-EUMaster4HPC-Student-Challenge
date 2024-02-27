#!/bin/bash

# Load modules
module load intel
module load OpenMPI

# Compile the code with MPI and OpenMP
mpicxx -O2 -fopenmp src/conjugate_gradients.cpp -o conjugate_gradients

# Set the number of OpenMP threads
export OMP_NUM_THREADS=4

#Create the input files
time ./random_spd_system.sh 20000 io/matrix.bin io/rhs.bin

# Run the executable with MPI
time mpirun -n 2 --oversubscribe ./conjugate_gradients io/matrix.bin io/rhs.bin io/sol.bin
