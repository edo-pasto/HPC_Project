# **OpenMP, MPI and CUDA Parallelization of Recursive Functions**

## **Final Project - High Performance Computing - A.Y. 2022/2023** 

### - Edoardo Pastorino, 5169595

### - Mahyar Sadeghi Garjan, 5283082



## Introduction:

 The aim of the project is to try to implement and explain a possible approach to parallelize recursive function, based on already existent experiments found online, with the usage of three of the most diffused interfaces for exploiting parallelism inside the code of a programming language. These API are `OpenMP`, `MPI` and `CUDA`. All these three parallelization methods are implemented, in our work, inside program in C language, due to the fact that are well supported by the language and by the relative compilers.



## Project Structure:

```

.

├── fibonacci

│   └── fibonacci.c

├── README.md

├── fibonacci_OpenMP

│   ├── fibonacci_par1.c

│   ├── fibonacci_par2.c

│   └── fibonacci_par3.c

├── fibonacci_MPI

│   ├── fib.c

│   ├── MakeFile

│   └── mpi-fib-start.c

├── fibonacci_MPI_OMP

│   ├── omp_mpi.c

│   └── omp_mpi_v2.c

└── fibonacci_CUDA

    ├── fibonacciCUDA.cu

    └── fibonacciCUDA._NoRecursivecu

```

- **`fibonacci directory`**, contains the basic version of the fibonacci recursive method (*`fibonacci.c`*), without any kind of parallelization.

- **`fibonacci_OpenMP directory`**, contains the three versions of fibonacci with OpenMp multi-threading parallelism (*`fibonacci_par1.c`, `fibonacci_par2.c`* and *`fibonacci_par3.c`*).

- **`fibonacci_MPI directory`**, contains the files used for the multi-processes approach with MPI interface, the entry point of the configuration (*`mpi-fib-start.c`*) and the fibonacci program executed in the different processes (*`fib.c`*).

- **`fibonacci_MPI_OMP directory`**, contains the two versions of the integration between MPI and OMP (*omp_mpi.c* and *omp_mpi_v2.c*).

- **`Fibonacci_CUDA directory`**, contains the recursive version (*`fibonacciCUDA.cu`*) and the non recursive version (*`fibonacciCUDA_NoRecursive.cu`*) of fibonacci, using the GPU computation and parallelism.



## General Instruction:

For compiling the `Basic` version of fibonacci:
```
gcc -std=c99 fibonacci.c -o heat.out
```
For compiling the three `OpenMP` fibonacci's versions:
```
icc -O2 -qopenmp -xHost fibonacci_par3.c -o fibonacci_par1.out
```
```
icc -O2 -qopenmp -xHost fibonacci_par3.c -o fibonacci_par2.out
```
```
icc -O2 -qopenmp -xHost fibonacci_par3.c -o fibonacci_par3.out
```
For compiling the files of `MPI` fibonacci's version:
```
mpiicc -o mpi-fib-start mpi2-fib-start.c
```
```
mpiicc -o fib fib.c
```
For executing the MPI fibonacci’s versions:
```
mpirun -np 1 mpi-fib-start 10
```
For compiling the two `MPI + OMP` fibonacci's versions:
```
mpiicc -qopenmp -xHost omp_mpi_v2.c -o omp_mpi.out
```
```
mpiicc -qopenmp -xHost omp_mpi_v2.c -o omp_mpi_v2.out
```
For executing the two `MPI + OMP` fibonacci's versions:
```
mpiexec -hostfile machinefile.txt -perhost 1 -np 8 ./omp_mpi.out 45
```
```
mpiexec -hostfile machinefile.txt -perhost 1 -np 8 ./omp_mpi_v2.out 45
```
For compiling the two `CUDA` fibonacci’s versions:
```
nvcc -o fibonacciCUDA.out fibonacciCUDA.cu 
```
```
nvcc -o fibonacciCUDA_NoRecursive.out fibonacciCUDA_NoRecursive.cu 
```

