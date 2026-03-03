# Gaussian Primitives for Fast SVR

Gaussian-based Slice-to-Volume Reconstruction (GSVR) for fetal brain MRI. Represents a 3D volume as a set of learnable Gaussian primitives and reconstructs high-resolution isotropic volumes from thick-slice 2D acquisitions.


## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on A100)
- PyTorch >= 2.0
- FAISS with GPU support (`faiss-gpu-cu12`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/config_subjects_real.yaml
```

## Configuration

All settings are in `configs/config_subjects_real.yaml`. Key sections:

| Section | Key | Description |
|---|---|---|
| `data.subjects` | `input_stacks` | Paths to input NIfTI stacks |
| `data.subjects` | `input_masks` | Optional mask files (set to `[]` for auto-masking) |
| `model` | `num_gaussians` | Number of Gaussian primitives (default: 100,000) |
| `model` | `init_type` | `content_adaptive` or `random` |
| `flags` | `use_motion_correction` | Enable per-slice motion correction |
| `flags` | `use_psf` | Enable PSF modelling |
| `flags` | `use_slice_scaling` | Enable per-slice intensity scaling |
| `flags` | `use_slice_uncertainty` | Enable aleatoric uncertainty weighting |
| `flags` | `use_masks` | `True` = mask background; `False` = train on all voxels |
| `training` | `max_epochs` | Number of training epochs |
| `training` | `checkpoint_every` | Save checkpoint every N epochs |
| `reconstruction` | `spacing` | Output voxel spacing (e.g., `[1, 1, 1]`) |
| `reconstruction` | `slice_thickness` | Per-stack slice thickness |

## Docker

Build and push the Docker image:

```bash
bash build.sh
```

Submit a training job (Run:ai):

```bash
bash run.sh
```

## Project Structure

```
├── model.py   # GSVR model, training loop, data loading
├── train.py                     # CLI entry point
├── configs/
│   └── config_subjects_real.yaml
├── Dockerfile
├── build.sh                     # Docker build script
├── run.sh                       # Run:ai job submission script
└── requirements.txt
```
