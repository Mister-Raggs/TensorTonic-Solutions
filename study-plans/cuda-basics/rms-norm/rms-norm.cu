#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // Write code here
    int row = blockIdx.x;

    if (row >= M){
        return;
    }
    float rms = 0.0;

    for (int i = 0; i < N; i++){
        rms += input[row*N + i] * input[row*N + i];
    }
    rms = sqrtf((rms / N) + eps);

    for (int j = 0; j < N; j++){
        output[row*N + j] = input[row*N + j] * gamma[j] / rms; 
    }
    return;
    
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}