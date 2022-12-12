#include "stdio.h" // printf
#include "stdlib.h" // malloc and rand for instance. Rand not thread sa$
#include "time.h"   // time(0) to get random seed
#include "math.h"  // sine and cosine
// #include <omp.h>

int fibonacci(int number);

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci");
        return 0;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("The input number for fibonacci is: %s\n", argv[1]);


    // printf("Max available threads = %d\n", maxthreads);
    // printf("Total threads used = %d\n",nthreads);

    // double start_time = omp_get_wtime();
    cudaEventRecord(start, 0);

    int res = fibonacci(atoi(argv[1]));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // double run_time = omp_get_wtime() - start_time;
    printf("Fibonacci computation in %f seconds\n",elapsedTime/1000);
    printf("The fibonacci of %s is %d\n", argv[1], res);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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