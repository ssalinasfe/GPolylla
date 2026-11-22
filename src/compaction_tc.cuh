#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>

#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

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
    int *d_scanned = NULL;
    gpuErrchk( cudaMalloc(&d_scanned, sizeof(int) * n) );

    // 2. Perform CUB Inclusive Sum on the bit vector
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    gpuErrchk( cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_scanned, n) );
    gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    gpuErrchk( cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_bit_vector, d_scanned, n) );

    // 3. Extract the total number of valid items from the last element
    int total_elements = 0;
    gpuErrchk( cudaMemcpy(&total_elements, d_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost) );
    
    // Copy the total elements to the d_num pointer (which resides on the device)
    gpuErrchk( cudaMemcpy(d_num, &total_elements, sizeof(INDEX), cudaMemcpyHostToDevice) );

    // 4. Scatter kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scatter_seed_edges<<<blocks, threads>>>(d_out, d_scanned, d_bit_vector, n);
    
    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(d_temp_storage);
    cudaFree(d_scanned);
}
