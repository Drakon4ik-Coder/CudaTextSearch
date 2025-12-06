#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>

// CUDA kernel: one thread per potential start index, block-level reduction of matches.
__global__ void count_kernel(const char* text, int n, const char* token, int tlen, int* outCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + tlen > n) return; // guard: cannot fit token here

    bool match = true;
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

// Host-side wrapper to be called from main later.
int gpu_count_token(const std::vector<char>& text, const std::string& token)
{
    if (token.empty()) return 0;
}
