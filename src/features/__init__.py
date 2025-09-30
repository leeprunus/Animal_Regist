"""
DINO Features Module

Contains classes and functions for DINO visual feature extraction and matching from 3D models.
"""

from .dino_extractor import DinoFeatureExtractor, process_all_models
from .matching import DinoMatcher

__all__ = [
    'DinoFeatureExtractor', 
    'process_all_models',
    'DinoMatcher'
]
