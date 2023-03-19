#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 32

// CUDA kernel to add two matrices using shared memory
__global__ void matrixAddition(float *A, float *B, float *C, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int index = row * N + col;

    float sum = 0.0;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < N && i * TILE_SIZE + tx < N) {
            sA[ty][tx] = A[row * N + i * TILE_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0;
        }

        if (i * TILE_SIZE + ty < N && col < N) {
            sB[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            sum += sA[ty][j] * sB[j][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[index] = sum;
    }
}

int main() {
    int N = 6400;

    // allocate host memory
    float *A, *B, *C;
    A = (float *) malloc(N * N * sizeof(float));
    B = (float *) malloc(N * N * sizeof(float));
    C = (float *) malloc(N * N * sizeof(float));

    // initialize matrices with random numbers
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float) rand() / (float) RAND_MAX;
        B[i] = (float) rand() / (float) RAND_MAX;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, N * N * sizeof(float));
    cudaMalloc((void **) &d_B, N * N * sizeof(float));
    cudaMalloc((void **) &d_C, N * N * sizeof(float));

    // copy input matrices to device memory
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // measure execution time for different block sizes
    cudaEvent_t start, stop;
    float elapsedTime;

    for (int block_size = 4; block_size <= 32; block_size += 4) {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y -1)/ dimBlock.y);

cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

matrixAddition<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime, start, stop);

printf("Block size: (%d, %d) - Execution time: %f ms\n", block_size, block_size, elapsedTime);

// copy output matrix from device to host memory
cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// free device memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

// free host memory
free(A);
free(B);
free(C);

return 0;
}
