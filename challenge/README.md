# Conjugate Gradient Solver for Meluxina

## Compile the code
To compile the code on Meluxina you need to load the following modules:

```bash
module load intel
module load CMake
```

Then, you can compile the code using the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

To compile just one of the tests, you can replace the `make` command with the name of the test you want to compile. For example, to compile the `test_CG_CPU_MPI_OMP` test, you can use the following command:

```bash
make test_CG_CPU_MPI_OMP.out
```

## Running the tests
The test_CG_CPU_MPI_OMP test can be run in two different modes: **File mode** and **Generate mode**.

1. **File Mode**: In this mode, you need to specify the input matrix file, the input right-hand side file, the output solution file, the maximum iterations, and the relative error.

Here is an example of how to execute the program in this mode:

```bash
srun ./test_CG_CPU_MPI_OMP_file.out -A <matrix_file> -b <rhs_file> -o <output_file> -i <max_iterations> -e <relative_error>
```

If any of the parameters are not provided, the program will use the following default values:
- **matrix_file**: The input matrix file. The default value is **"io/matrix.bin"**.
- **rhs_file**: The input right-hand side file. The default value is **"io/rhs.bin"**.
- **output_file**: The output solution file. The default value is **"io/sol.bin"**.
- **max_iterations**: The maximum number of iterations. The default value is **1000**.
- **relative_error**: The relative error. The default value is **1e-8**.


2. **Generate Mode**:  In this mode, you only need to provide the size of the matrix. The program will **generate** a matrix and a right-hand side of the specified size, solve the system, and write the solution to a file.

Here is an example of how to execute the program in this mode:

```bash
srun ./test_CG_CPU_MPI_OMP_gen.out -s <matrix_size> -o <output_file> -i <max_iterations> -e <relative_error>
```
use the following commands:


