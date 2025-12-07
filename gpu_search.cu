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
    if (threadIdx.x == 0 && sdata[0] > 0)
        atomicAdd(outCount, sdata[0]);
}

namespace {
inline void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}
}

struct GpuSearchContext {
    char* d_text = nullptr;
    int text_len = 0;
    int* d_count = nullptr;
    int threads = 256;

    ~GpuSearchContext() { release(); }

    void init(const std::vector<char>& text)
    {
        release();
        text_len = static_cast<int>(text.size());
        if (text_len == 0) return;
        checkCuda(cudaMalloc(&d_text, text_len * sizeof(char)), "cudaMalloc d_text");
        checkCuda(cudaMemcpy(d_text, text.data(), text_len * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy d_text");
        checkCuda(cudaMalloc(&d_count, sizeof(int)), "cudaMalloc d_count");
    }

    void release()
    {
        if (d_text) cudaFree(d_text);
        if (d_count) cudaFree(d_count);
        d_text = nullptr;
        d_count = nullptr;
        text_len = 0;
    }

    // Returns count; if kernel_ms is provided, fills it with kernel time in ms.
    int count_token(const std::string& token, double* kernel_ms = nullptr)
    {
        if (!d_text || text_len == 0 || token.empty()) return 0;
        const int tlen = static_cast<int>(token.size());

        char* d_token = nullptr;
        checkCuda(cudaMalloc(&d_token, tlen * sizeof(char)), "cudaMalloc d_token");
        checkCuda(cudaMemcpy(d_token, token.data(), tlen * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy d_token");
        checkCuda(cudaMemset(d_count, 0, sizeof(int)), "cudaMemset d_count");

        const int blocks = (text_len + threads - 1) / threads;
        const size_t sharedBytes = threads * sizeof(int);

        cudaEvent_t start, stop;
        if (kernel_ms) {
            checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
            checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
            checkCuda(cudaEventRecord(start), "cudaEventRecord start");
        }

        count_kernel<<<blocks, threads, sharedBytes>>>(d_text, text_len, d_token, tlen, d_count);
        checkCuda(cudaGetLastError(), "count_kernel launch");

        if (kernel_ms) {
            checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
            checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
            float ms = 0.f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            *kernel_ms = static_cast<double>(ms);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        } else {
            checkCuda(cudaDeviceSynchronize(), "count_kernel sync");
        }

        int result = 0;
        checkCuda(cudaMemcpy(&result, d_count, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy result");
        cudaFree(d_token);
        return result;
    }
};

// Global context helpers for reuse.
static GpuSearchContext g_ctx;
static bool g_ctx_initialized = false;

void gpu_init_text(const std::vector<char>& text)
{
    g_ctx.init(text);
    g_ctx_initialized = true;
}

void gpu_set_block_size(int threads)
{
    if (threads > 0) {
        g_ctx.threads = threads;
    }
}

int gpu_get_block_size()
{
    return g_ctx.threads;
}

int gpu_count_token_reuse(const std::string& token, double* kernel_ms)
{
    if (!g_ctx_initialized)
        throw std::runtime_error("GPU context not initialized. Call gpu_init_text first.");
    return g_ctx.count_token(token, kernel_ms);
}

// Simple wrapper to preserve previous API; reinitializes text each call.
int gpu_count_token(const std::vector<char>& text, const std::string& token)
{
    gpu_init_text(text);
    return gpu_count_token_reuse(token, nullptr);
}
