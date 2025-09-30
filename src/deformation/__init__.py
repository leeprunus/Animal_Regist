"""
Deformation Module

Functions for mesh deformation including Linear Blend Skinning and AutoRig processing.
"""

from .lbs import lbs_mesh_deformation
from .autorig import run_autorig

__all__ = [
    'lbs_mesh_deformation',
    'run_autorig'
]