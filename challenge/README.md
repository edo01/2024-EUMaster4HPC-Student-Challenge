# Please follow the steps for running the code on MeluXina

# Conjugate gradient method



## Intruduction

Hello, I am John Doe, and I am a student on the University of Science. This is my final project in the programming class.

It is a program which solves a system of equations using the conjugate gradient method. It is an iterative solver, which needs only matrix-vector multiplications and some vector operations to find a solution. To find more information about the algorithm, read your notes from past Linear algebra classes, or see e.g. [this Wikipedia page](https://en.wikipedia.org/wiki/Conjugate_gradient_method).



## Description of the project

The main program in this project is `conjugate_gradients`, which solves the system. It loads and input dense matrix in row-major format and a right-hand-side from given binary files, performs the conjugate gradient iterations until convergence, and then writes the found solution to a given output binary file. A symmetric positive definite matrix and a right-hand-side can be generated using the `random_spd_system.sh` script and program.

Inorder to test your code on MeluXina, please use the interactive node (for quick checking)
```
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```


To make the program work, I need to execute this command first
```
module load intel
```

Create a directory for the input and output files
```
mkdir io
```

To compile the program, I use
```
icpx -O2 src/conjugate_gradients.cpp -o conjugate_gradients
```

To generate a random SPD system of 10000 equations and unknowns, use e.g.
```
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
```

To then solve the system, use
```
./conjugate_gradients io/matrix.bin io/rhs.bin io/sol.bin
```
It takes almost a minute to run this program.



## Task

Unfortunately, my programming teacher was not happy with this project. He said that the conjugate gradient solver is very slow. But I have no idea how to make it faster.

__Will you help me make the program as fast as possible?__

You can modify my code as much as you like. But the main functionality of the program has to stay the same - solve a system of equations using the conjugate gradient method.
