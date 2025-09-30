"""
AutoRig Module

Blender-based automatic rigging and deformation for 3D meshes.
This module contains Blender scripts that must be run as subprocesses.
"""

import os

# Path to the Blender AutoRig script
AUTORIG_SCRIPT = os.path.join(os.path.dirname(__file__), 'blender_rig.py')

__all__ = ['AUTORIG_SCRIPT']
