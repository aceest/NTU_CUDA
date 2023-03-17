
__global__ void matrixAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int index = i * N + j;
        C[index] = A[index] + B[index];
    }
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch kernel on the device
    matrixAdd<<<dim3((N + optimalBlock.x - 1) / optimalBlock.x, (N + optimalBlock.y - 1) / optimalBlock.y), optimalBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();




eSt aCe
keep going
scss
Copy code
// calculate the maximum absolute error
float maxError = 0.0f;
for (int i = 0; i < N; ++i) {
    float error = fabs( (1.0/h_A[i] + 1.0/h_B[i]) - h_C[i] );
    maxError = fmaxf(error, maxError);
}

// Release host memory
free(h_A);
free(h_B);
free(h_C);
free(h_D);

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









