// Vector Dot Product A.B
// compile with the following command:
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* d_A;   // device vectors
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecDot(const float* A, const float* B, float* C, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp += A[i]*B[i];
        i += blockDim.x*gridDim.x;  
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib)
        cache[cacheIndex] += cache[cacheIndex + ib]; 

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0)
      C[blockIdx.x] = cache[0];

}

// Host code

int main(void)
{
    printf("\n");
    printf("Vector Dot product with multiple GPUs \n");
    int N, NGPU, cpu_thread_id=0;
    int *Dev; 
    long mem = 1024*1024*1024;     // 4 Giga for float data type.

    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int)*NGPU);

    int numDev = 0;
    printf("GPU device number: ");
    for(int i = 0; i < NGPU; i++) {
      scanf("%d", &Dev[i]);
      printf("%d ",Dev[i]);
      numDev++;
      if(getchar() == '\n') break;
    }
    printf("\n");
    if(numDev != NGPU) {
      fprintf(stderr,"Should input %d GPU device numbers\n", NGPU);
      exit(1);
    }

    printf("Enter the size of the vectors: ");
    scanf("%d", &N);        
    printf("%d\n", N);        
    if (3*N > mem) {
        printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
        exit(1);
    }
    long size = N*sizeof(float);

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if(threadsPerBlock > 1024) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(1);
    }
    int blocksPerGrid = (N + threadsPerBlock*NGPU - 1) / (threadsPerBlock*NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(1);
    }

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    if (! h_A || ! h_B || ! h_C) {
	      printf("!!! Not enough memory.\n");
	      exit(1);
    }
    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);
    // Initialize h_C to zero
    memset(h_C, 0, size);

    
    //float Intime,gputime,Outime;
     // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    omp_set_num_threads(NGPU);
    double* h_G = (double*)malloc(sizeof(double) * NGPU);

    #pragma omp parallel private(cpu_thread_id)
    {
	    float *d_A, *d_B, *d_C;
	    cpu_thread_id = omp_get_thread_num();
	    cudaSetDevice(Dev[cpu_thread_id]);
      //cudaSetDevice(cpu_thread_id);

     // Create the timer for each thread
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   

    // start the timer
    cudaEventRecord(start,0);


    // Allocate vectors in device memory
	  cudaMalloc((void**)&d_A, size/NGPU);
	  cudaMalloc((void**)&d_B, size/NGPU);
	  cudaMalloc((void**)&d_C, size/NGPU);

    // Copy vectors from host memory to device memory
	  cudaMemcpy(d_A, h_A+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);
	  cudaMemcpy(d_B, h_B+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);
	  //#pragma omp barrier

  
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    VecDot<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N/NGPU);
	  cudaDeviceSynchronize();
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",2*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result of each block in host memory

    // start the timer
    cudaEventRecord(start,0);

    
    cudaMemcpy(h_C+N/NGPU*cpu_thread_id, d_C, size/NGPU, cudaMemcpyDeviceToHost);
	
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   // Store the result in h_G array
    h_G[cpu_thread_id] = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) 
        h_G[cpu_thread_id] += (double) h_C[i];
     // Destroy the timer for each thread
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Reset CUDA device
        cudaDeviceReset();
     // Final sum reduction
    double h_G_total = 0.0;
    for (int i = 0; i < NGPU; i++) {
        h_G_total += h_G[i];
    }
    // stop the timer
    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    // to compute the reference solution

    double h_D=0.0;       
    for(int i = 0; i < N; i++) 
      h_D += (double) h_A[i]*h_B[i];
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",2*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    double diff = fabs( (h_D - h_G_total)/h_D );
    printf("|(h_G - h_D)/h_D|=%20.15e\n",diff);
    printf("h_G =%20.15e\n",h_G_total);
    printf("h_D =%20.15e\n",h_D);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();
    }
}



// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}






