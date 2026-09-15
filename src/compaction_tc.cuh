#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

typedef int INDEX;

__global__ void scatter_seed_edges(INDEX *out, int *scanned, int *bits, INDEX n_elements) {
    INDEX idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (bits[idx] == 1) {
            out[scanned[idx] - 1] = idx;
        }
    }
}

// Native CUB-based compaction for seed edges.
// This completely replaces the experimental/broken WMMA snippet with a robust, high-performance CUB scan + scatter.
void compaction_tc_scan(INDEX *d_out, INDEX *d_num, int *d_bit_vector, INDEX n) {
    // 1. Allocate memory for the inclusive scan output
    int *d_scanned;
    cudaMalloc(&d_scanned, sizeof(int) * n);

    // 2. Perform CUB Inclusive Sum on the bit vector
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_scanned, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_scanned, n);

    // 3. Extract the total number of valid items from the last element
    int total_elements = 0;
    cudaMemcpy(&total_elements, d_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Copy the total elements to the d_num pointer (which resides on the device)
    cudaMemcpy(d_num, &total_elements, sizeof(INDEX), cudaMemcpyHostToDevice);

    // 4. Scatter kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scatter_seed_edges<<<blocks, threads>>>(d_out, d_scanned, d_bit_vector, n);
    
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage);
    cudaFree(d_scanned);
}
