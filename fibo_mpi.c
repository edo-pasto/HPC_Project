#include "stdio.h" 
#include "stdlib.h" 
#include "time.h"   
#include "math.h" 
#include <mpi.h>

int fibonacci(int number);

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci!\n");
        return 0;
    }

    printf("The input number for fibonacci is: %s\n", argv[1]);

    int maxthreads = omp_get_max_threads();
    int nthreads = omp_get_num_threads();
    printf("Max available threads = %d\n", maxthreads);
    printf("Total threads used = %d\n",nthreads);

    double start_time = omp_get_wtime();

    int res = fibonacci(atoi(argv[1]));

    double run_time = omp_get_wtime() - start_time;
    printf("Total DFTW computation in %f seconds\n",run_time);

    printf("The fibonacci of %s is %d\n", argv[1], res);
}

int fibonacci(int number){

    if(number == 0){
        return 0;
    }else if( number == 1){
        return 1;
    }

     int x = fibonacci(number - 1);
     int y = fibonacci(number - 2);
    
    return x + y;

}