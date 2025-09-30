#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements = read_requirements('requirements.txt')

# Extract version from __init__.py or set manually
__version__ = "1.0.0"

setup(
    name="Animal_Regist",
    version=__version__,
    author="leeprunus",
    author_email="leeprunus@gmail.com",
    description="Training-free Non-Rigid Registration of Articulated Animal Bodies via Vision Features and Anatomical Priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leeprunus/Animal_Regist",
    project_urls={
        "Bug Tracker": "https://github.com/leeprunus/Animal_Regist/issues",
        "Documentation": "https://github.com/leeprunus/Animal_Regist/blob/main/README.md",
        "Source Code": "https://github.com/leeprunus/Animal_Regist",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.10.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "animal_regist=correspondence:main",
        ],
    },
    include_package_data=True,
    package_data={
        "Animal_Regist": [
            "config/*.yaml",
            "src/autorig/*.py",
        ],
    },
    keywords=[
        "3d correspondence",
        "animal models", 
        "computer vision",
        "geometry processing",
        "mesh alignment",
        "keypoint detection",
        "DINO features",
        "ARAP registration",
    ],
    zip_safe=False,
)
