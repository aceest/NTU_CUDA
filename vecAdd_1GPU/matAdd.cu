// Matrix addition: C = A + B.
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>



// Variables

float* h_A;   // host matrix
float* h_B;
float* h_C;
float* h_D;
float* d_A;   // device matrix
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void matAdd(float* A, float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
	    int index = i*N + j;
	    C[index] = A[index] + B[index];
    //	C[i][j] = A[i][j] + B[i][j];
    }
    __syncthreads();
}

// Host code

int main( ){
    

    int gid;   // GPU_ID

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU_ID: ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    int mem = 1024*1024*1024;     // Giga    
    int N;

    printf("Enter the size of the vectors: ");
    scanf("%d",&N);        
    printf("%d\n",N);        
    if( N*N > mem ) {     // each real number (float) takes 4 bytes
	 printf("The size of these 3 vectors cannot be fitted into 6 Gbyte\n");
	 exit(2);
	 }

    // Allocate input matrix h_A and h_B in host memory

    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));
    
    // Initialize the input matrix with random numbers
    srand(0);
    RandomInit(h_A, N*N);
    RandomInit(h_B, N*N);

    // Set the sizes of threads and blocks

    int threadsPerBlock;
    
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    
    
   
   
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 numBlocks((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    // create the timer
    cudaevent_t start, stop;
    cudaeventcreate(&start);
    cudaeventcreate(&stop);

    // start the timer
    cudaeventrecord(start,0);

    // allocate vectors in device memory

    cudamalloc((void**)&d_A,  N * N * sizeof(float));
    cudaMalloc((void**)&d_B,  N * N * sizeof(float));
    cudaMalloc((void**)&d_C,  N * N * sizeof(float));

    // Copy matrix from host memory to device memory

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

 
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    matAdd<<<numBlocks, block>>>(d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",3*N*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);
    
    h_D = (float*)malloc( N * N * sizeof(float));       // to compute the reference solution
    for (int i = 0; i < N; ++i){ 
	    for (int j = 0; j <N; ++j){
        	h_D[i][j] = h_A[i][j] + h_B[i][j];
	    }
    }
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (int i = 0; i < N; ++i) {
	    for ( int j = 0; j < N; ++j){
      		diff = abs(h_D[i][j] - h_C[i][j]);
      		sum += diff*diff; 
	    }
//      if(diff > 1.0e-15) { 
//        printf("i=%d, h_D=%15.10e, h_C=%15.10e \n", i, h_D[i], h_C[i]);
//      }
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);

    cudaDeviceReset();
}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for(int i = 0; i< n; i++)
        data[i] = rand() / (float)RAND_MAX;
}



