#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N){
        return;
    }

    if (input[i] >= 0){
        output[i] = input[i];
    }
    else {
        output[i] = alpha * input[i];
    }
    return;
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}