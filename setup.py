from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dtw_cuda",
    version="0.1",
    author="Leyang Xue",
    description="CUDA-accelerated Dynamic Time Warping (DTW) for PyTorch",
    ext_modules=[
        CUDAExtension(
            name="dtw_cuda",
            sources=["csrc/dtw_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
