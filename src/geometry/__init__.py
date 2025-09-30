"""
Geometry Module

Geometric computation utilities for mesh processing and analysis.
"""

from .hks import HeatKernelSignatureGPU, TipDetector

__all__ = [
    'HeatKernelSignatureGPU',
    'TipDetector'
]
