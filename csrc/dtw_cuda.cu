// dtw_cuda.cu + CPU version + all-pairs CUDA (shared memory, rolling buffer)
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define INF 1e20

// -----------------------------
// CPU DTW
// -----------------------------
torch::Tensor dtw_cpu(torch::Tensor seq1, torch::Tensor seq2) {
    int len1 = seq1.size(0);
    int len2 = seq2.size(0);

    auto dtw_matrix = torch::empty({len1 + 1, len2 + 1}, torch::TensorOptions().dtype(torch::kFloat32));
    dtw_matrix.fill_(INF);
    dtw_matrix.index_put_({0, 0}, 0.0f);

    auto a = seq1.accessor<float, 1>();
    auto b = seq2.accessor<float, 1>();
    auto m = dtw_matrix.accessor<float, 2>();

    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            float cost = std::abs(a[i - 1] - b[j - 1]);
            float min_cost = std::min({m[i - 1][j], m[i][j - 1], m[i - 1][j - 1]});
            m[i][j] = cost + min_cost;
        }
    }

    return torch::tensor({m[len1][len2]}, torch::TensorOptions().dtype(torch::kFloat32));
}

// -----------------------------
// All-pairs DTW with shared memory rolling buffer (2D sequences)
// -----------------------------
__global__ void dtw_all_pairs_rolling_kernel(
    const float* __restrict__ sequences,  // [N, L]
    float* __restrict__ output,           // [N, N]
    int N, int L) {

    int i = blockIdx.y;
    int j = blockIdx.x;

    if (i >= N || j >= N || j < i) return;

    const float* seq1 = sequences + i * L;
    const float* seq2 = sequences + j * L;

    extern __shared__ float buffer[];
    float* prev = buffer;
    float* curr = buffer + (L + 1);

    for (int k = threadIdx.x; k <= L; k += blockDim.x) {
        prev[k] = INF;
        curr[k] = INF;
    }
    if (threadIdx.x == 0) prev[0] = 0.0f;
    __syncthreads();

    for (int x = 1; x <= L; ++x) {
        if (threadIdx.x == 0) curr[0] = INF;
        __syncthreads();

        for (int y = 1 + threadIdx.x; y <= L; y += blockDim.x) {
            float cost = fabsf(seq1[x - 1] - seq2[y - 1]);
            float min_cost = fminf(prev[y], fminf(curr[y - 1], prev[y - 1]));
            curr[y] = cost + min_cost;
        }
        __syncthreads();

        float* temp = prev;
        prev = curr;
        curr = temp;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[i * N + j] = prev[L];
        output[j * N + i] = prev[L];  // symmetric
    }
}

torch::Tensor dtw_all_pairs_cuda(torch::Tensor sequences) {
    TORCH_CHECK(sequences.dim() == 2, "Input must be 2D (N, L)");

    int N = sequences.size(0);
    int L = sequences.size(1);

    cudaSetDevice(sequences.get_device());
    TORCH_CHECK(N > 0 && L > 0, "Input tensor must be non-empty");
    TORCH_CHECK(sequences.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(sequences.dtype() == torch::kFloat32, "Input tensor must be float32");

    auto options = sequences.options().dtype(torch::kFloat32);
    auto output = torch::empty({N, N}, options);

    dim3 blocks(N, N);
    dim3 threads(32);
    size_t shared_mem_size = 2 * (L + 1) * sizeof(float);

    dtw_all_pairs_rolling_kernel<<<blocks, threads, shared_mem_size>>>(
        sequences.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dtw_cpu", &dtw_cpu, "DTW CPU kernel");
    m.def("dtw_cuda", &dtw_all_pairs_cuda, "All-pairs DTW CUDA with rolling buffer");
}
