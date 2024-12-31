// Main program to test vector addition
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "vector_add.cuh"

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)


////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////


std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

////////////////////////////////////////////////////////////////////////////////
///          RUN BENCHMARKS
////////////////////////////////////////////////////////////////////////////////


struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    bool save_result;
};

float compute_rrmse(const std::vector<float>& ref, const std::vector<float>& test) {
    float mse = 0.0f, ref_norm = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = ref[i] - test[i];
        mse += diff * diff;
        ref_norm += ref[i] * ref[i];
    }
    mse /= ref.size();
    ref_norm /= ref.size();
    return std::sqrt(mse / ref_norm);
}

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs
) {
    for (auto config: configs) {
        auto size_i = config.size_i;

        // load a,b,c as cpp vectors
        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i);
        auto a = read_data(path_prefix + "_a.bin", size_i);
        auto b = read_data(path_prefix + "_b.bin", size_i);
        auto c = read_data(path_prefix + "_c.bin", size_i);


        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_i * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(a_gpu, a.data(), size_i * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_gpu, b.data(), size_i * sizeof(float), cudaMemcpyHostToDevice));
        
        ////////////////////////////////////////////////////////////////
        //              RUNNING IMPL
        ////////////////////////////////////////////////////////////////

        Impl::run(size_i, a_gpu, b_gpu, c_gpu);
        std::vector<float> c_out_host(size_i);
        CUDA_CHECK(cudaMemcpy(c_out_host.data(), c_gpu, size_i * sizeof(float), cudaMemcpyDeviceToHost));


        float rel_rmse = compute_rrmse(c, c_out_host);
        printf("  size %4d:\n", size_i);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("running %s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, true}});
}


struct VectorAddV1 {
    constexpr static char const *name = "vector_add_v1";
    static void
    run(int32_t size_i,
        float const *a,
        float const *b,
        float *c
        ) {
        vector_add::launch_vector_add(a, b, c, size_i);
    }
};


int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    printf("test data dir: %s\n", test_data_dir.c_str());

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<VectorAddV1>(test_data_dir, saved_results);
}
