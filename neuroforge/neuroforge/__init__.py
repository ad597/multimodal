"""
NeuroForge - Custom AI Infrastructure

A high-performance AI compute framework with custom GPU kernels optimized 
for multimodal attention mechanisms and embodied AI applications.

Features:
- Custom CUDA kernels for attention mechanisms
- Multimodal fusion operations
- Memory-efficient implementations
- Hardware acceleration for large-scale RL simulations
"""

__version__ = "0.1.0"
__author__ = "AI Infrastructure Team"

# Core modules
from .core.attention_ops import (
    OptimizedMultiHeadAttention,
    CrossModalAttention,
    BenchmarkSuite
)

__all__ = [
    "OptimizedMultiHeadAttention",
    "CrossModalAttention", 
    "BenchmarkSuite",
    "__version__"
]

# Check for custom kernels
try:
    import neuroforge_kernels
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

print(f"NeuroForge v{__version__} - Custom kernels: {'✅ Available' if KERNELS_AVAILABLE else '⚠️  Not available'}")
