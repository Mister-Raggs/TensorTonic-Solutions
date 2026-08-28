#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N){
        return;
    }
    atomicAdd(result, input[i]);
    return;
    // iterate over i, incrementing by 1 until all elements covered. if i reaches a factor of N, increase Block Idx by 1 and start counting again?
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
