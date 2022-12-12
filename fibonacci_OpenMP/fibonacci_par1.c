#include "stdio.h" // printf
#include "stdlib.h" // malloc and rand for instance. Rand not thread safe!
#include "time.h"   // time(0) to get random seed
#include "math.h"  // sine and cosine
#include <omp.h>

int fibonacci(int number);

int main(int argc, char* argv[]){
    int res;

    if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci!\n");
        return 0;
    }

    printf("The input number for fibonacci is: %s\n", argv[1]);

    int maxthreads = omp_get_max_threads();
    printf("Max available threads = %d\n", maxthreads);
    int nthreads_par;
    int nthreads_sin;

    double start_time = omp_get_wtime();
    #pragma omp parallel num_threads(128)
    {
        nthreads_par = omp_get_num_threads();
        #pragma omp single
        {
            nthreads_sin = omp_get_num_threads();
            res = fibonacci(atoi(argv[1]));
        }
        
    }
    double run_time = omp_get_wtime() - start_time;
    printf("Total DFTW computation in %f seconds\n",run_time);

    printf("Total threads used after parallel declaration = %d\n",nthreads_par);
    printf("Total threads used after single declaration = %d\n",nthreads_sin);


    printf("The fibonacci of %s is %d\n", argv[1], res);
}

int fibonacci(int number){

    if(number == 0){
        return 0;
    }else if( number == 1){
        return 1;
    }
    int x, y;
    
    #pragma omp task shared(x)
    {
        // printf("Total threads used inside fib = %d\n", omp_get_num_threads());
        x = fibonacci(number - 1);
    }
    #pragma omp task shared(y)
    {
        y = fibonacci(number - 2);
    }

    #pragma omp taskwait
    
    return x + y;

}