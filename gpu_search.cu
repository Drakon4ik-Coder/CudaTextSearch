#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

// CUDA kernel: one thread per potential start index, block-level reduction of matches.
__global__ void count_kernel(const char* text, int n, const char* token, int tlen, int* outCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool match = true;
    if (i + tlen > n) {
        match = false; // cannot fit token here
    } else {
        #pragma unroll 8
        for (int k = 0; k < tlen; ++k) {
            if (text[i + k] != token[k]) { match = false; break; }
        }

        // Boundary checks: neighbors must be non-letters (or out of range).
        if (match) {
            char prefix = (i == 0) ? ' ' : text[i - 1];
            char suffix = (i + tlen >= n) ? ' ' : text[i + tlen];
            if ((prefix >= 'a' && prefix <= 'z') || (suffix >= 'a' && suffix <= 'z'))
                match = false;
        }
    }

    // Block-level reduction of match flags to reduce atomic pressure.
    extern __shared__ int sdata[];
    sdata[threadIdx.x] = match ? 1 : 0;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(outCount, sdata[0]);
}

// Host-side wrapper: copies data to GPU, launches kernel, returns match count.
int gpu_count_token(const std::vector<char>& text, const std::string& token)
{
    if (token.empty() || text.empty()) return 0;

    auto check = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << what << ": " << cudaGetErrorString(err);
            throw std::runtime_error(oss.str());
        }
    };

    const int n = static_cast<int>(text.size());
    const int tlen = static_cast<int>(token.size());

    char* d_text = nullptr;
    char* d_token = nullptr;
    int* d_count = nullptr;

    check(cudaMalloc(&d_text, n * sizeof(char)), "cudaMalloc d_text");
    check(cudaMalloc(&d_token, tlen * sizeof(char)), "cudaMalloc d_token");
    check(cudaMalloc(&d_count, sizeof(int)), "cudaMalloc d_count");

    check(cudaMemcpy(d_text, text.data(), n * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy text");
    check(cudaMemcpy(d_token, token.data(), tlen * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy token");
    check(cudaMemset(d_count, 0, sizeof(int)), "cudaMemset d_count");

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const size_t sharedBytes = threads * sizeof(int);
    count_kernel<<<blocks, threads, sharedBytes>>>(d_text, n, d_token, tlen, d_count);
    check(cudaGetLastError(), "count_kernel launch");
    check(cudaDeviceSynchronize(), "count_kernel sync");

    int result = 0;
    check(cudaMemcpy(&result, d_count, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy result");

    cudaFree(d_text);
    cudaFree(d_token);
    cudaFree(d_count);
    return result;
}
