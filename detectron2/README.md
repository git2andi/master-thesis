This folder contains training and evaluation code for Faster R-CNN models using Detectron2.

## Setup

Detectron2 must be installed following the official instructions:

[https://detectron2.readthedocs.io/en/latest/tutorials/install.html](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

### Example Conda Environment

```bash
conda create -n detectron2 python=3.10 -y
conda activate detectron2
```
Install a compatible version of PyTorch and torchvision before installing Detectron2, then follow the instructions.

## File Placement

After installing Detectron2:

* Copy all configuration files from this repository’s `config/` folder into the corresponding `configs/` directory of the Detectron2 GitHub repository.
* Copy `custom_train.py` into the root directory of the Detectron2 GitHub repository.

## Usage

### Training

```bash
python custom_train.py \
  --num-gpus 2 \
  --dist-url auto \
  --config-file path/to/config.yaml
```

### Evaluation

```bash
python custom_train.py \
  --num-gpus 1 \
  --dist-url auto \
  --config-file path/to/config.yaml \
  --eval-only \
  MODEL.WEIGHTS path/to/best/model.pth
```
