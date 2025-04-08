import torch
import dtw_cuda
from dtaidistance import dtw
import time
import numpy as np

x = torch.randn(128, 672, device="cuda") 

start = time.perf_counter()
dist_cpu = dtw.distance_matrix_fast(x.cpu().numpy().astype(np.double), use_c=True)
dist_cpu = dist_cpu.astype(np.float32)
end = time.perf_counter()
print("CPU DTW time:", end - start)

start = time.perf_counter()
dist_gpu = dtw_cuda.dtw_cuda(x)
end = time.perf_counter()
print("CUDA DTW time:", end - start)


dist_cpu = torch.tensor(dist_cpu).to("cuda")

dist_cpu = dist_cpu / torch.max(dist_cpu)
dist_gpu = dist_gpu / torch.max(dist_gpu)


# Check if the distances are equal
is_equal = torch.allclose(dist_cpu, dist_gpu)
print("Are the distances equal?", is_equal)

max_diff = torch.max(torch.abs(dist_cpu - dist_gpu))
print("Max difference:", max_diff.item())
print("CPU Distance:", dist_cpu)
print("CUDA Distance:", dist_gpu)