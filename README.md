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

### CLI Config Overrides

Any config value can be overridden from the command line using dot-notation, without editing the YAML file. The format is `--section.key value`:

```bash
# Change batch size and number of epochs
python train.py --config configs/config_subjects_real.yaml \
    --training.batch_size 500000 \
    --training.max_epochs 100

# Toggle flags
python train.py --config configs/config_subjects_real.yaml --flags.use_masks False

# Change model parameters
python train.py --config configs/config_subjects_real.yaml --model.num_gaussians 200000

# Override nested keys like learning rates
python train.py --config configs/config_subjects_real.yaml --training.learning_rates.position 0.01

# Combine multiple overrides
python train.py --config configs/config_subjects_real.yaml \
    --training.max_epochs 50 \
    --model.num_gaussians 200000 \
    --flags.use_psf False \
    --training.learning_rates.position 0.01
```

Values are automatically cast to match the type of the existing config value (int, float, bool, or string).

### Custom Experiment Name

```bash
python train.py --config configs/config_subjects_real.yaml --exp_name my_experiment
```

## Configuration

All settings are in `configs/config_subjects_real.yaml`. Key sections:

| Section | Key | Description |
|---|---|---|
| `data.subjects` | `input_stacks` | Paths to input NIfTI stacks |
| `data.subjects` | `input_masks` | Optional mask files (set to `[]` for auto-masking) |
| `model` | `num_gaussians` | Number of Gaussian primitives (default: 100,000) |
| `model` | `init_type` | `content_adaptive` or `random` |
| `model` | `init_lambda` | Edge vs uniform init balance (0.0 = edges only, 1.0 = uniform) |
| `flags` | `use_motion_correction` | Enable per-slice motion correction |
| `flags` | `use_psf` | Enable PSF modelling |
| `flags` | `use_slice_scaling` | Enable per-slice intensity scaling |
| `flags` | `use_slice_uncertainty` | Enable aleatoric uncertainty weighting |
| `flags` | `use_masks` | `True` = mask background; `False` = train on all voxels |
| `training` | `max_epochs` | Number of training epochs (default: 500) |
| `training` | `batch_size` | Mini-batch size for voxel processing (default: 1,000,000) |
| `training` | `learning_rates.*` | Per-parameter learning rates (position, scaling, rotation, etc.) |
| `reconstruction` | `spacing` | Output voxel spacing (e.g., `[1, 1, 1]`) |
| `reconstruction` | `slice_thickness` | Per-stack slice thickness |
| `preprocessing` | `bias_field_correction` | Enable N4 bias field correction |
| `preprocessing` | `denoise` | Enable denoising |

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
├── model.py                       # GSVR model, training loop, data loading
├── train.py                       # CLI entry point
├── configs/
│   └── config_subjects_real.yaml
├── Dockerfile
├── build.sh                       # Docker build script
├── run.sh                         # Run:ai job submission script
└── requirements.txt
```
