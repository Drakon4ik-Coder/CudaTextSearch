#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <cctype>
#include <chrono>

int gpu_count_token(const std::vector<char>& text, const std::string& token);
void gpu_init_text(const std::vector<char>& text);
int gpu_count_token_reuse(const std::string& token, double* kernel_ms = nullptr);

std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i< int(data.size()); ++i)
    {
        // test 1: does this match the token?
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the prefix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

int main()
{
    // Example chosen file
    const char * filepath = "dataset/shakespeare.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Example word list
    const char * words[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};

    // Initialize GPU context once with the text to avoid per-word copies.
    gpu_init_text(file_data);
    // Warm up CUDA context and JIT; discard this timing.
    gpu_count_token_reuse("warmup", nullptr);

    for(const char * word : words)
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        int cpu_occurrences = calc_token_occurrences(file_data, word);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        int gpu_occurrences = -1;
        double gpu_ms_host = -1.0;
        double gpu_ms_kernel = -1.0;
        try {
            auto gpu_start = std::chrono::high_resolution_clock::now();
            gpu_occurrences = gpu_count_token_reuse(std::string(word), &gpu_ms_kernel);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            gpu_ms_host = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        } catch (const std::exception& ex) {
            std::cerr << "GPU count failed for word '" << word << "': " << ex.what() << std::endl;
        }

        std::cout << "Word: " << word
                  << " | CPU: " << cpu_occurrences << " (" << cpu_ms << " ms)"
                  << " | GPU: " << gpu_occurrences << " (host: " << gpu_ms_host << " ms, kernel: " << gpu_ms_kernel << " ms)"
                  << " | Diff: " << (cpu_occurrences - gpu_occurrences)
                  << std::endl;
    }

    return 0;
}
