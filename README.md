# `MEt3R`: Measuring Multi-View Consistency in Generated Images [CVPR 2025] 
<a href="https://mohammadasim98.github.io">Mohammad Asim</a><sup>1</sup>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a><sup>1</sup>, <a href="https://wimmerth.github.io">Thomas Wimmer</a><sup>1, 2</sup>, <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/">Bernt Schiele</a><sup>1</sup>,  <a href="https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html">Jan Eric Lenssen</a><sup>1</sup>

*<sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus, <sup>2</sup>ETH Zurich*

<h4 align="left">
<a href="https://geometric-rl.mpi-inf.mpg.de/met3r/">Project Page</a>
</h4>

### `TL;DR: A differentiable metric to measure multi-view consistency between an image pair`.

---

### 📣 News

- **15.04.2025** - Updates:
  - Added optical flow-based warping backbone using [`RAFT`](https://arxiv.org/abs/2003.12039).
  - Added `psnr`, `ssim`, `lpips`, `rmse`, and `mse` metrics on warped RGB images instead of feature maps.
  - Added `nearest`, `bilinear` and `bicubic` upsampling methods.
  - Refactored codebase structure.  
- **26.02.2025** - Accepted to [`CVPR 2025`](https://cvpr.thecvf.com/) 🎉!
- **10.01.2025** - Initial code releases.

---

## 🔍 Method Overview 
<div align="center">
  <img src="assets/method_overview.jpg" width="800"/>
</div>

**MEt3R** evaluates the consistency between images $\mathbf{I}_1$ and $\mathbf{I}_2$. Given such a pair, we apply **DUSt3R** to obtain dense 3D point maps $\mathbf{X}_1$ and $\mathbf{X}_2$. These point maps are used to project upscaled **DINO** features $\mathbf{F}_1$, $\mathbf{F}_2$ into the coordinate frame of $\mathbf{I}_1$, via unprojecting and rendering. We compare the resulting feature maps $\hat{\mathbf{F}}_1$ and $\hat{\mathbf{F}}_2$ in pixel space to obtain similarity $S(\mathbf{I}_1,\mathbf{I}_2)$.

---

## 📋 Contents
- [📓 Abstract](#-abstract)
- [📌 Dependencies](#-dependencies)
- [🛠️ Quick Setup](#️-quick-setup)
- [📣 Example Usage](#-example-usage)
- [⚙️ Benchmarking Models](#-benchmarking-models)
- [👷 Manual Install](#-manual-install)
- [📘 Citation](#-citation)

---

## 📓 Abstract
We introduce **MEt3R**, a metric for multi-view consistency in generated images. Large-scale generative models for multi-view image generation are rapidly advancing the field of 3D inference from sparse observations. However, due to the nature of generative modeling, traditional reconstruction metrics are not suitable to measure the quality of generated outputs and metrics that are independent of the sampling procedure are desperately needed. In this work, we specifically address the aspect of consistency between generated multi-view images. Our approach uses **DUSt3R** to obtain dense 3D reconstructions from image pairs, which are used to warp image contents from one view into the other. Feature maps of these images are compared to obtain a similarity score that is invariant to view-dependent effects. Using **MEt3R**, we evaluate the consistency of a large set of previous methods for novel view and video generation, including our open, multi-view latent diffusion model.

---

## 📌 Dependencies

    - Python >= 3.6
    - PyTorch >= 2.1.0
    - CUDA >= 11.3
    - PyTorch3D >= 0.7.5
    - FeatUp >= 0.1.1

Tested with *CUDA 11.8*, *PyTorch 2.4.1*, *Python 3.10*

---

## 🛠️ Quick Setup

Install **MEt3R** via:

```bash
pip install git+https://github.com/mohammadasim98/met3r
```

---

## 💡 Example Usage

```python
import torch
from met3r import MEt3R

IMG_SIZE = 256

metric = MEt3R(
    img_size=IMG_SIZE,
    use_norm=True,
    backbone="mast3r",
    feature_backbone="dino16",
    feature_backbone_weights="mhamilton723/FeatUp",
    upsampler="featup",
    distance="cosine",
    freeze=True,
).cuda()

inputs = torch.randn((10, 2, 3, IMG_SIZE, IMG_SIZE)).cuda().clip(-1, 1)

score, *_ = metric(images=inputs)
print(score.mean().item())

torch.cuda.empty_cache()
```

More in `example.ipynb`

---

## ⚙️ Benchmarking Models

**MEt3R** now supports benchmarking performance of generative models on multi-view consistency.

### 🔧 Model Installation
Install and generate outputs using the following repos:
- **SV4D / SV4D2**: [https://github.com/Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)
- **ViFiGen**: [https://github.com/xiexh20/ViFiGen](https://github.com/xiexh20/ViFiGen)

### 🚀 Running Evaluation

After generating model outputs, run the evaluation using the provided `eval.py` script:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MEt3R model")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--use_norm", action="store_true", help="Use normalization")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images or videos")
    parser.add_argument("--feature_backbone", type=str, default="dino16", help="Feature backbone for the model")
    parser.add_argument("--backbone", type=str, default="dust3r", help="Backbone architecture for the model")
    parser.add_argument("--data_type", type=str, choices=["image", "video"], default="video", help="Type of data to evaluate")
    parser.add_argument("--model_name", type=str, default="sv4d2", help="Name of the model to evaluate")
```

Run via command line:

```bash
python eval.py \
    --img_size 256 \
    --batch_size 2 \
    --data_dir path/to/generated/outputs \
    --feature_backbone dino16 \
    --backbone dust3r \
    --data_type video \
    --model_name sv4d2 \
    --use_norm
```

---

## 👷 Manual Install

```bash
pip install -r requirements.txt
```

Install **FeatUp**:
```bash
pip install git+https://github.com/mhamilton723/FeatUp
```

Install **PyTorch3D**:
```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git
```

Install **DUSt3R** submodule:
```bash
git submodule update --init --recursive
```

---

## 📘 Citation

```bibtex
@inproceedings{asim24met3r,
  title = {MEt3R: Measuring Multi-View Consistency in Generated Images},
  author = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
  booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
  year = {2024},
}
```
