# Training-free Non-Rigid Registration of Articulated Animal Bodies via Vision Features and Anatomical Priors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Installation

### Prerequisites

- **Conda/Miniconda**: Download from https://docs.conda.io/en/latest/miniconda.html
- **NVIDIA GPU with CUDA 11.8** (recommended)
- **Blender**: `sudo apt install blender` (Linux) or download from blender.org

### Manual Installation (Tested on Ubuntu)

```bash
# Clone the repository
git clone https://github.com/leeprunus/Animal_Regist.git
cd Animal_Regist

# 1. Create environment
conda create -n AniCorres python=3.10 -y
conda activate AniCorres

# 2. Install PyTorch with CUDA
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Install core dependencies (CRITICAL: NumPy must be <2.0)
pip install "numpy>=1.26.4,<2.0"
pip install scipy==1.11.4 scikit-learn matplotlib "pandas>=2.0.3,<2.1"

# 4. Install 3D processing
conda install -c conda-forge open3d=0.18.0 vtk -y
pip install trimesh

# 5. Install computer vision
pip install opencv-python==4.9.0.80
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmpose==1.3.2 mmdet==3.3.0 mmengine==0.10.7

# 6. Install additional packages
pip install transformers==4.35.2 tokenizers==0.15.2
pip install chumpy==0.70 polyscope==2.4.0 iopath==0.1.10
pip install json-tricks munkres xtcocotools pycocotools shapely terminaltables
pip install triton==2.0.0 pyyaml tqdm requests cython setuptools cmake ninja safetensors huggingface-hub

# 7. Install Blender (required for mesh deformation)
sudo apt update
sudo apt install blender
```

### Verification

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import numpy as np; print(f'NumPy: {np.__version__}')
import open3d as o3d; print('Open3D: OK')
import mmcv; print(f'MMCV: {mmcv.__version__}')
import mmpose; print('MMPose: OK')
print('Installation successful!')
"
```


## Usage

### Basic Usage

Process two 3D animal models to establish dense correspondences:

```bash
python correspondence.py source_model.obj target_model.obj
```

Supports multiple 3D formats (.obj, .ply, .stl, .glb, .gltf, .dae, .3ds, .x3d):
```bash
python correspondence.py model1.glb model2.ply
```

### Mesh Simplification Options

**⚠️ Important Note:** By default, the pipeline automatically simplifies input meshes to improve processing speed and avoid memory issues. The simplified meshes (not the originals) are used throughout the pipeline.

- **Default behavior**: Meshes are simplified to `max_vertices` (configurable in `config/pipeline.yaml`, default: 5000 vertices)
- **Process original meshes**: Use `--no-simplify` to work with full-resolution meshes (slower, requires more memory)

```

**Recommendation**: For fast processing, keep `max_vertices: 5000` in the config. This provides a good balance between detail preservation and computational efficiency.

### Custom Configuration

Use custom configuration file:

```bash
python correspondence.py --config custom_config.yaml source_model.obj target_model.obj
```

### Output Structure

Each run creates a timestamped result directory under `result/`:

```
result/
└── source_to_target_YYYYMMDD_HHMMSS/
    ├── correspondences.json          # Final correspondence results
    ├── meshes/                       # Actual meshes used in pipeline
    │   ├── source_processed.obj      # Processed source mesh
    │   └── target_processed.obj      # Processed target mesh
    └── [intermediate_files...]       # Other pipeline outputs
```

**Key Points:**
- **`correspondences.json`**: Contains the final vertex correspondences between meshes
- **`meshes/`**: Contains the actual meshes that were processed (simplified if `--no-simplify` was not used)
- The processed meshes in `meshes/` folder are self-contained for this specific run

### Correspondence JSON Format

The `correspondences.json` file has the following structure:

```json
{
  "correspondences": {
    "0": 156,      // Source vertex 0 maps to target vertex 156
    "1": 248,      // Source vertex 1 maps to target vertex 248
    "2": 891,      // etc.
    ...
  },
  "metadata": {
    "total_correspondences": 1000,
    "method": "mge_evaluation",
    "mge": 0.032574                   // Mean Geodesic Error (only works for ground truth mesh)
  }
}
```

- **Keys** in `correspondences`: Source mesh vertex indices (as strings)
- **Values**: Corresponding target mesh vertex indices (as integers)
- **MGE**: Mean Geodesic Error measuring correspondence quality (lower is better)

## Project Structure

```
anicorres/
├── config/
│   └── pipeline.yaml          # Main configuration
├── src/
│   ├── alignment/             # ARAP alignment algorithms
│   ├── deformation/           # Mesh deformation methods
│   ├── features/              # DINO feature extraction
│   ├── geometry/              # Geometric processing
│   ├── keypoints/             # 3D keypoint detection
│   ├── mesh/                  # Mesh processing utilities
│   ├── preprocessing/         # Pipeline preprocessing
│   └── utils/                 # Common utilities
├── correspondence.py          # Main pipeline script
├── requirements.txt          # Python dependencies
└── environment.yml           # Conda environment configuration
```

## Citation

If you use this method in your research, please cite:

```bibtex
@article{training_free_registration2024,
  title={Training-free Non-Rigid Registration of Articulated Animal Bodies via Vision Features and Anatomical Priors},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.