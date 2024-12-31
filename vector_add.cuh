namespace vector_add {

constexpr uint32_t n_threads = 256;

__global__ void add_vectors(const float* a, const float* b, float* c, int size) {
    int idx = threadIdx.x + blockIdx.x * n_threads;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    uint32_t n_blocks = (size + n_threads - 1) / n_threads;
    add_vectors<<<n_blocks, n_threads>>>(a, b, c, size);


}


} // namespace vector_add