"""
Setup script for NeuroForge - Custom AI Infrastructure
Compiles CUDA kernels and installs the package.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
import os
import sys

# Check if CUDA is available
def check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Get CUDA paths
def get_cuda_paths():
    cuda_available = check_cuda()
    if not cuda_available:
        return None, None, None
    
    # Try to find CUDA installation
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Try common CUDA installation paths
        for path in ['/usr/local/cuda', '/opt/cuda', '/usr/local/cuda-12.0', '/usr/local/cuda-11.8']:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if cuda_home is None:
        print("Warning: CUDA not found. Kernels will not be compiled.")
        return None, None, None
    
    include_dirs = [
        os.path.join(cuda_home, 'include'),
        pybind11.get_include(),
        torch.utils.cpp_extension.CUDA_HOME + '/include' if torch.utils.cpp_extension.CUDA_HOME else '',
    ]
    
    library_dirs = [
        os.path.join(cuda_home, 'lib64'),
        os.path.join(cuda_home, 'lib'),
    ]
    
    libraries = ['cudart', 'cublas', 'curand', 'cublasLt']
    
    return include_dirs, library_dirs, libraries

# Define the extension module
def get_extensions():
    include_dirs, library_dirs, libraries = get_cuda_paths()
    
    if include_dirs is None:
        print("CUDA not available. Installing CPU-only version.")
        return []
    
    # Define the CUDA extension
    ext_modules = [
        Pybind11Extension(
            "neuroforge_kernels",
            sources=[
                "kernels/attention_cuda.cu",
                "kernels/python_bindings.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language='c++',
            cxx_std=17,
            define_macros=[
                ('VERSION_INFO', '"dev"'),
                ('TORCH_EXTENSION_NAME', 'neuroforge_kernels'),
                ('_GLIBCXX_USE_CXX11_ABI', '1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0'),
            ],
        )
    ]
    
    return ext_modules

# Setup configuration
setup(
    name="neuroforge",
    version="0.1.0",
    author="AI Infrastructure Team",
    description="Custom AI Infrastructure with optimized GPU kernels for multimodal attention",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "benchmark": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "tqdm>=4.62.0",
        ],
    },
    packages=[
        "neuroforge",
        "neuroforge.core",
        "neuroforge.kernels",
        "neuroforge.benchmarks",
        "neuroforge.examples",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
