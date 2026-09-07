#include <cuda_runtime.h>

__global__ void l1_normalize_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Stride loop: one block, 256 threads, N may be much larger.
    float partial = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        partial += fabsf(input[i]);
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Safe to read: the loop's last barrier already ran, and nobody
    // writes sdata after it. All 256 threads see the finished total.
    float denom = sdata[0];

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = input[i] / denom;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    l1_normalize_kernel<<<1, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}