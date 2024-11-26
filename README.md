
# MET3R: Measuring Multi-View Consistency in Generated Images.
<h5 align="center">

[![arXiv]()]()
[![project page]()]()
</h5>

## Abstract
We introduce MET3R, a metric for multi-view consistency in generated images. Large-scale generative models for multi-view image generation are rapidly advancing the field of 3D inference from sparse observations. However, due to the nature of generative modeling, traditional reconstruction metrics are not suitable to measure the quality of generated outputs and metrics that are independent of the sampling procedure are desperately needed. In this work, we specifically address the aspect of consistency between generated multi-view images, which can be evaluated independently of the specific scene. MET3R uses DUSt3R to obtain dense 3D reconstructions from image pairs in a feed-forward manner, which are used to warp image contents from one view into the other. Then, feature maps of these images are compared to obtain a similarity score that is invariant to view-dependent effects. Using MET3R, we evaluate the consistency of a large set of previous methods and our own, open, multi-view latent diffusion model.

## 📌 Dependencies

- **Python >= 3.6**
- **torch >= 2.1.0**
- **torchvision >= 0.16.0**
- **CUDA >= 11.3**

Tested with *CUDA 11.8*, *PyTorch 2.4.1*, *Python 3.10*

## 🛠️ Quick Setup
Simply install MET3R using the following command inside a bash terminal assuming prequisites are aleady installed and working.
```bash
pip install git+https://github.com/mohammadasim98/met3r
```

## 👷 Manual Install

Additionally MET3R can also be installed manually in a local development environment. 
### Install Prerequisites
```bash
pip install -r requirements.txt
```
### Installing FeatUp
MET3R relies on FeatUp to generate high resolution feature maps for the input images. Install FeatUp using the following command. 

```bash
pip install git+https://github.com/mhamilton723/FeatUp
```
Refer to [FeatUp](https://github.com/mhamilton723/FeatUp) for more details.

### Installing Pytorch3D
MET3R requires Pytorch3D to perform point projection and rasterization. Install it via the following command.  
```bash 
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
In case of issues related to installing and building Pytorch3D, refer to [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details. 

### Installing DUSt3R
At the core of MET3R lies [DUSt3R](https://github.com/naver/dust3r) which is used to generate the 3D point maps for feature unprojection and rasterization. Due to LICENSE issues, we adopt DUSt3R as a submodule which can be downloaded as follows.
```bash
git submodule update --init --recursive
```

## Example Usage

Simply import and use MET3R in your codebase as follows.

```python
import torch
from met3r import MET3R

metric = MET3R().cuda()

inputs = torch.randn((10, 2, 3, 256, 256)).cuda()
inputs = inputs.clip(-1, 1)

"""
Args:
    Inputs (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
    return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted), 
    return_projections (bool, False): Return projected feature maps

Return:
    score (float): MET3R score which consists of weighted mean of feature dissimlarity
    mask (bool[Tensor, "b c h w"])
"""

score, mask = metric(inputs)

# Should be between 0.3 - 0.35
print(score.mean().item())
```

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```

```