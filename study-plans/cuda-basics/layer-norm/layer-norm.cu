#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma,
                                  const float* beta, float* output,
                                  int M, int N, float eps) {
    int row = blockIdx.x;

    float mean = 0.0f;
    for (int col = 0; col < N; col++) {
        mean += input[row * N + col];
    }
    mean /= N;

    float var = 0.0f;
    for (int col = 0; col < N; col++) {
        float d = input[row * N + col] - mean;
        var += d * d;
    }
    var /= N;

    float inv_std = rsqrtf(var + eps);

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        output[row * N + col] =
            (input[row * N + col] - mean) * inv_std * gamma[col] + beta[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta,
                      float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}