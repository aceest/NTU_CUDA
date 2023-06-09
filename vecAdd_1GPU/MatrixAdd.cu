// <atrix addition: C = A + B.
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* h_D;
float* d_A;   // device vectors
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void MatAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < N && j < N){
	    int index = i*N +j;
       	    C[index] = A[index] + B[index];
    }
    __syncthreads();
}

int main() {
    const int N = 6400; // size of the matrix
    const int blockSizes[6][2] = {{4, 4}, {8, 8}, {10, 10}, {16, 16}, {20, 20}, {32, 32}};

    // Allocate memory for matrices A, B, and C on host
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];
    float* D = new float[N * N];

    // Initialize matrices A and B with random values
    srand(0);
    randomInit(A, N * N);
    randomInit(B, N * N);

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Find optimal block size
    int maxThreads = 0;
    dim3 optimalBlock(1, 1);
    for (int i = 0; i < 6; ++i) {
        dim3 blockSize(blockSizes[i][0], blockSizes[i][1]);
        dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        int threads = numBlocks.x * numBlocks.y * blockSize.x * blockSize.y;
        if (threads > maxThreads && threads <= 1024) {
            maxThreads = threads;
            optimalBlock = blockSize;
        }
    }

    printf("Optimal block size: (%d, %d)\n", optimalBlock.x, optimalBlock.y);

    printf("Matrix Addition: C = A + B\n");
    int mem = 1024*1024*1024;     // Giga    
    int N;

    printf("Enter the size of the matrix: ");
    scanf("%d",&N);        
    printf("%d\n",N);        
    if( N*N > mem ) {     // each real number (float) takes 4 bytes
      printf("The size of these 3 vectors cannot be fitted into 6 Gbyte\n");
      exit(2);
    }
    long size = N * N * sizeof(float);

    // Launch kernel on the device
    matrixAdd<<<dim3((N + optimalBlock.x - 1) / optimalBlock.x, (N + optimalBlock.y - 1) / optimalBlock.y), optimalBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();


// calculate the maximum absolute error
float maxError = 0.0f;
for (int i = 0; i < N; ++i) {
    float error = fabs( (1.0/h_A[i] + 1.0/h_B[i]) - h_C[i] );
    maxError = fmaxf(error, maxError);
}

    RandomInit(h_A, N*N);
    RandomInit(h_B, N*N);

// Exit and clean up
if (maxError < 1e-5) {
    printf("Test PASSED\n");
}
else {
    printf("Test FAILED\n");
}
return 0;
}

// Randomly initialize an array
void RandomInit(float* data, int size)
{
for (int i = 0; i < size; ++i)
data[i] = rand() / (float)RAND_MAX;
}

    int threadsPerBlock;
loop:
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if( threadsPerBlock > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      goto loop;
    }
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    printf("The number of blocks is %d\n", blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      goto loop;
    }
     dim3 blockSize(threadsPerBlock, threadsPerBlock);
     dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    MatAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",3*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
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

    h_D = (float*)malloc(size);       // to compute the reference solution
    for (int i = 0; i < N; ++i) {
	    for (int j = 0; j < N; ++j){
		    int index = i*N + j;
		    h_D[index] = h_A[index] + h_B[index]; 
	    }
    }
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));
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
	    	int index = i*N + j;
		    diff = abs(h_D[index] - h_C[index]);
	    	sum += diff*diff;
	    }
      
//      if(diff > 1.0e-15) { 
//        printf("i=%d, h_D=%15.10e, h_C=%15.10e \n", i, h_D[i], h_C[i]);
//      }
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);






