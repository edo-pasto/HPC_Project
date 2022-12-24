#include "stdio.h" 
#include "stdlib.h" 
#include "time.h"  
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <memory> 
__device__ void fibonacci(int number, int* res);
__global__ void fibonacciParent(int* number, int* res);

int initial_fibonacci_run(int fib, int currentDepth, int targetDepth);

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    printf("%d   %d   %d   %s\n", devProp.major, devProp.minor, mp, devProp.name);
    switch (devProp.major){
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: //GeForce GTX 1650
        cores = mp * 64;
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

int calc_CUDA_Fibonacci(int number){
    //get number of cores

    // The number of Graphics cards in this computer
    int deviceCount = 0;
    // The Index Device we're going to use (Default 0)
    int currentDevice = 0;
    // The number of CUDA cores on the device being used
    int CUDACoreCount = 0;
    // How deep we need to go into recursion to get the optimal number of threads.
    int depth = 0;
    // Error logging
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0){
        printf("Cannot find any CUDA device.");
        exit(EXIT_FAILURE);
    }

    // Get the information on the current device
    cudaSetDevice(currentDevice);
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, currentDevice);

    // Get the number of CUDA Cores on the current device
    CUDACoreCount = getSPcores(deviceProperties);

    // How deep we need to go into the recursion before we spawn threads
    depth = floor(log2((double)CUDACoreCount));
    // printf("   \n%d\n   ",depth);
    int x = initial_fibonacci_run(number, 0, depth);   
    return x;


}


int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci!\n");
        return 0;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("The input number for fibonacci is: %s\n", argv[1]);

    int number = atoi(argv[1]);
    int res;
    float elapsedTime;
    
    cudaEventRecord(start, 0);

    printf("The fibonacci of %s is %d\n", argv[1], calc_CUDA_Fibonacci(number));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    printf("Fibonacci computation in %f seconds\n",elapsedTime/1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int initial_fibonacci_run(int fib, int currentDepth, int targetDepth){
    if (fib <= 1)
        return fib;

    if (currentDepth < targetDepth)
        return initial_fibonacci_run(fib - 1, currentDepth++, targetDepth) + initial_fibonacci_run(fib - 2, currentDepth++, targetDepth);

    int *d_fib, *d_result;

    int size = sizeof(int);
    cudaMalloc((void**)&d_fib, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_fib, &fib, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, 0, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = (fib + threadsPerBlock - 1) / threadsPerBlock;
    fibonacciParent<<<blocksPerGrid, threadsPerBlock>>>(d_fib, d_result);
    int result = 0;
    cudaMemcpy(&result, d_result, size, cudaMemcpyDeviceToHost);    

    cudaFree(d_fib);
    cudaFree(d_result);
    return result;
}

__global__ void fibonacciParent(int* number, int* res){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < *number){
        fibonacci(*number, res);
    }
   
    return;

}

__device__ void fibonacci(int number, int* res){

    if (number <= 1){
        *res += number;
        return;
    }

    int i =  blockIdx.x * blockDim.x + threadIdx.x;

        fibonacci(number - 1, res);
        fibonacci(number - 2, res);
}
