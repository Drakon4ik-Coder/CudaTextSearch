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
void gpu_set_block_size(int threads);
int gpu_get_block_size();

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

struct Stats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

Stats compute_stats(const std::vector<double>& samples)
{
    Stats s{};
    if (samples.empty()) return s;
    s.min_ms = *std::min_element(samples.begin(), samples.end());
    s.max_ms = *std::max_element(samples.begin(), samples.end());
    double sum = 0.0;
    for (double v : samples) sum += v;
    s.avg_ms = sum / samples.size();
    return s;
}

bool run_sanity_tests()
{
    struct Case { std::string text; std::vector<std::pair<std::string,int>> queries; };
    std::vector<Case> cases = {
        {"ana ana", {{"ana", 2}, {"an", 0}}},
        {"banana!", {{"banana", 1}, {"ana", 0}, {"ban", 0}}},
        {"a man, a plan, a canal", {{"a", 3}, {"man", 1}, {"plan", 1}, {"canal", 1}}},
        {"edge", {{"edge", 1}, {"ed", 0}, {"dge", 0}}},
        {" aaa aaa", {{"aaa", 2}, {"aa", 0}}},
    };

    bool all_pass = true;
    for (const auto& c : cases) {
        std::vector<char> buf(c.text.begin(), c.text.end());
        std::transform(buf.begin(), buf.end(), buf.begin(), [](char ch){ return std::tolower(static_cast<unsigned char>(ch)); });
        gpu_init_text(buf);
        for (const auto& q : c.queries) {
            int cpu = calc_token_occurrences(buf, q.first.c_str());
            int gpu = gpu_count_token_reuse(q.first);
            if (cpu != q.second || gpu != cpu) {
                std::cerr << "Sanity test failed on text \"" << c.text << "\" token \"" << q.first
                          << "\" expected " << q.second << " cpu " << cpu << " gpu " << gpu << "\n";
                all_pass = false;
            }
        }
    }
    return all_pass;
}

int main(int argc, char** argv)
{
    // Optional CLI: argv[1]=iterations (default 10), argv[2]=block size (threads per block, default 256)
    int iterations = 10;
    if (argc > 1) iterations = std::max(1, std::stoi(argv[1]));
    if (argc > 2) {
        int bs = std::stoi(argv[2]);
        if (bs > 0) gpu_set_block_size(bs);
    }

    // Example chosen file
    const char * filepath = "dataset/beowulf.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Sanity tests on small synthetic strings
    if (!run_sanity_tests()) {
        std::cerr << "Sanity tests failed; aborting.\n";
        return -2;
    }

    // Example word list
    const char * words[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};

    // Initialize GPU context once with the text to avoid per-word copies.
    gpu_init_text(file_data);
    // Warm up CUDA context and JIT; discard this timing.
    gpu_count_token_reuse("warmup", nullptr);

    std::cout << "Iterations per word: " << iterations
              << " | GPU block size: " << gpu_get_block_size() << "\n";

    for(const char * word : words)
    {
        std::vector<double> cpu_samples;
        std::vector<double> gpu_host_samples;
        std::vector<double> gpu_kernel_samples;
        int cpu_occurrences = -1;
        int gpu_occurrences = -1;

        for (int it = 0; it < iterations; ++it) {
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_occurrences = calc_token_occurrences(file_data, word);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            cpu_samples.push_back(std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());
        }

        for (int it = 0; it < iterations; ++it) {
            double kernel_ms = -1.0;
            auto gpu_start = std::chrono::high_resolution_clock::now();
            gpu_occurrences = gpu_count_token_reuse(std::string(word), &kernel_ms);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            gpu_host_samples.push_back(std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count());
            gpu_kernel_samples.push_back(kernel_ms);
        }

        auto cpu_stats = compute_stats(cpu_samples);
        auto gpu_host_stats = compute_stats(gpu_host_samples);
        auto gpu_kernel_stats = compute_stats(gpu_kernel_samples);

        std::cout << "Word: " << word
                  << " | CPU count: " << cpu_occurrences
                  << " | GPU count: " << gpu_occurrences
                  << " | Diff: " << (cpu_occurrences - gpu_occurrences) << "\n";
        std::cout << "  CPU ms    (avg/min/max): " << cpu_stats.avg_ms << " / " << cpu_stats.min_ms << " / " << cpu_stats.max_ms << "\n";
        std::cout << "  GPU host  (avg/min/max): " << gpu_host_stats.avg_ms << " / " << gpu_host_stats.min_ms << " / " << gpu_host_stats.max_ms << "\n";
        std::cout << "  GPU kernel(avg/min/max): " << gpu_kernel_stats.avg_ms << " / " << gpu_kernel_stats.min_ms << " / " << gpu_kernel_stats.max_ms << "\n";
    }

    return 0;
}
