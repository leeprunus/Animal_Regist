"""
Animal_Regist Source Modules

Modular structure containing the functions needed by correspondence.py
"""

from . import keypoints
from . import mesh 
from . import features
from . import alignment
from . import evaluation
from . import utils
from . import geometry
from . import autorig
from . import deformation
from . import preprocessing
from . import pipeline

__version__ = "1.0.0"
__all__ = [
    'keypoints', 
    'mesh', 
    'features', 
    'alignment', 
    'evaluation',
    'utils',
    'geometry',
    'autorig',
    'deformation',
    'preprocessing',
    'pipeline'
]
