#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N){
        return;
    }
    float maxel = input[0];
    for (int j = 0; j < N; j++){ // calculate once per block but persist?
        if (input[j] > maxel){
            maxel = input[j];
        }
    }
    float denom = 0;
    for (int j = 0; j < N; j++){ 
        denom += expf(input[j] - maxel);
    }
    output[i] = (expf(input[i] - maxel)) / denom;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}