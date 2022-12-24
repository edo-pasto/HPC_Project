#include<stdio.h>
#include<math.h>
#include<cuda.h>
// #define N 10

__global__ void Fibonacci(double *ga, double *gb, double sqrt_five, double phi1, double phi2, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N)
	{
		gb[i] = (pow((double)phi1, ga[i]) - pow((double)phi2, ga[i])) / sqrt_five;
	}

}

int main(int argc, char *argv[])
{
	 if(argc != 2){
        printf("Error: You have to insert the input number for fibonacci!\n");
        return 0;
    }
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	float elapsedTime;

	int N = 1 + atoi(argv[1]);
	double ha[N], hb[N];// Host variable
	double *ga,*gb; //For GPU use

	double sqrt_five, phi1, phi2;
	sqrt_five = sqrt(5);
	phi1 = (sqrt_five + 1) / 2;
	phi2 = (sqrt_five - 1) / 2;


	// Initialize array on CPU
	for (int i = 0; i<N; i++)
	{
		ha[i] = i;		
	}

	//Allocate memory on GPU
	cudaMalloc((void**)&ga, N*sizeof(double));
	cudaMalloc((void**)&gb, N*sizeof(double));

	cudaEventRecord(start, 0);
	//Copy array from CPU to GPU
	cudaMemcpy(ga, ha, N*sizeof(double), cudaMemcpyHostToDevice);

	//Kernel launching
	Fibonacci <<<2, 128 >>>(ga, gb, sqrt_five, phi1, phi2, N);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	//Copy results from GPU to CPU
	cudaMemcpy(ha, gb, N*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    printf("GPU Version, Elapsed time (s) = %f\n",elapsedTime/1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	printf("Result of Fibonacci GPU of %d is: %lf\n", N-1, ha[N-1]);

	return 0;

}
