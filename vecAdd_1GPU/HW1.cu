// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 6400
#define BLOCK_SIZE 16

__global__ void matrix_addition(float *a, float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        int index = i * n + j;
        c[index] = a[index] + b[index];
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int matrix_size = N * N * sizeof(float);
    int i, j;
    int block_size_x, block_size_y;
    dim3 block_dim, grid_dim;
    clock_t start, end;
    double cpu_time_used, gpu_time_used;
    float random_number;

    // Allocate memory on the host
    a = (float*) malloc(matrix_size);
    b = (float*) malloc(matrix_size);
    c = (float*) malloc(matrix_size);

    // Allocate memory on the device
    cudaMalloc(&d_a, matrix_size);
    cudaMalloc(&d_b, matrix_size);
    cudaMalloc(&d_c, matrix_size);

    // Initialize matrices a and b with random numbers
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            random_number = (float) rand() / (float) RAND_MAX;
            a[i * N + j] = random_number;
            random_number = (float) rand() / (float) RAND_MAX;
            b[i * N + j] = random_number;
        }
    }

    // Copy data from host to device
    cudaMemcpy(d_a, a, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, matrix_size, cudaMemcpyHostToDevice);

    // Perform matrix addition on the GPU with different block sizes
    for (block_size_x = 4; block_size_x <= 32; block_size_x += 4) {
        block_size_y = BLOCK_SIZE / block_size_x;
        block_dim = dim3(block_size_x, block_size_y, 1);
        grid_dim = dim3((N + block_size_x - 1) / block_size_x, (N + block_size_y - 1) / block_size_y, 1);

        start = clock();
        matrix_addition<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        end = clock();

        gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Block size (%d, %d): %f seconds\n", block_size_x, block_size_y, gpu_time_used);
    }

    // Perform matrix addition on the CPU
    start = clock();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i * N + j] = a[i * N + j] + b[i * N + j];
        }
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER



