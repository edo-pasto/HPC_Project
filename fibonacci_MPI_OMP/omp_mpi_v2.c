#include <stdio.h>
#include <time.h>
#include<mpi.h>
#include<omp.h>

int parallel_fibonacci(int number);
int serial_fibonacci(int number);
int callOMP(int number, int rank);

int main(int argc, char* argv[]){
    
    int size, rank;
    int err;
    double time2, total;
    time_t time1 = clock();
    double start = omp_get_wtime();

	err=MPI_Init(&argc, &argv);
	err=MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	err=MPI_Comm_size(MPI_COMM_WORLD, &size);
    int final_res=0;
    int res0=0,res1=0,res2=0,res3=0,res4=0,res5=0,res6=0,res7=0;
    if (rank==0){
    if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci!\n");
        return 0;
    }
        res0=callOMP(atoi(argv[1])-3,rank);
        MPI_Recv(&res1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
        MPI_Recv(&res2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(&res3, 1, MPI_INT, 3, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(&res4, 1, MPI_INT, 4, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(&res5, 1, MPI_INT, 5, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(&res6, 1, MPI_INT, 6, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
        MPI_Recv(&res7, 1, MPI_INT, 7, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    final_res=res0+res1+res2+res3+res4+res5+res6+res7;
    // final_res= res0+ 3*res1+ 3*res2+ res3+ res4+ 3*res5+ 3*res6+ res7;
    }
    else if (rank==1 ){
        res1=3*callOMP(atoi(argv[1])-4,rank);
        MPI_Send(&res1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==2){
        res2=3*callOMP(atoi(argv[1])-5,rank);
         MPI_Send(&res2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==3){
        res3=callOMP(atoi(argv[1])-7,rank);
         MPI_Send(&res3, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==4){
        res4=callOMP(atoi(argv[1])-11,rank);
         MPI_Send(&res4, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==5){
        res5=3*callOMP(atoi(argv[1])-12,rank);
         MPI_Send(&res5, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==6){
        res6=3*callOMP(atoi(argv[1])-13,rank);
         MPI_Send(&res6, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank==7){
        res7=callOMP(atoi(argv[1])-14,rank);
         MPI_Send(&res7, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    err = MPI_Finalize();
    
    time2 = (clock() - time1) / (double) CLOCKS_PER_SEC;
   double runtimeOMP = omp_get_wtime() - start;
   total = total + runtimeOMP;

        if (rank==0)
           { printf("Final Fibo res for fibo(%d): %d\n",atoi(argv[1]), final_res);
            // printf("Elapsed time (s) = %.2lf\n", time2);
            // printf("Elapsed time HotSpot (s) = %.2lf\n\n", runtimeOMP);
            }
            


    return 0;
}




int callOMP(int number, int rank){
    
    int nthreads_par;
    int nthreads_sin;
    int res;
    #pragma omp parallel num_threads(128)
    {
        nthreads_par = omp_get_num_threads();
        #pragma omp single
        {
            nthreads_sin = omp_get_num_threads();
            res = parallel_fibonacci(number);
        }
        
    }

    printf("We are in rank%d and we are computing fibo(%d) \n   result:%d \n",rank,number,res);

    return res;

}



int parallel_fibonacci(int number){

    if(number == 0){
        return 0;
    }else if( number == 1){
        return 1;
    }
    if(number <= 30){
        return serial_fibonacci(number);
    }
    int x, y;
    
    #pragma omp task shared(x)
    {
        // printf("Total threads used inside fib = %d\n", omp_get_num_threads());
        x = parallel_fibonacci(number - 1);
    }
    #pragma omp task shared(y)
    {
        y = parallel_fibonacci(number - 2);
    }

    #pragma omp taskwait
    
    return x + y;

}

int serial_fibonacci(int number){

    if(number == 0){
        return 0;
    }else if( number == 1){
        return 1;
    }

     int x = serial_fibonacci(number - 1);
     int y = serial_fibonacci(number - 2);
    
    return x + y;

}