
# import os
# os.environ['TORCH_HOME'] = '/BS/grl-masim-data/work/torch_models'

import torch
from torch import Tensor
from pathlib import Path
from torch.nn import Module
from jaxtyping import Float, Bool
from typing import Union, Tuple
from einops import rearrange, repeat

# Load featup
from featup.util import norm, unnorm

# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

# Load DUSt3R model
import met3r.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo 
from dust3r.utils.geometry import xy_grid


def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()

class MET3R(Module):

    def __init__(
        self, 
        img_size: int | None = 256, 
        use_norm: bool=True,
        feat_backbone: str="dino16",
        featup_weights: str | Path ="mhamilton723/FeatUp",
        dust3r_weights: str | Path ="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        **kwargs
    ) -> None:
        """Initialize MET3R

        Args:
            img_size (int, optional): Image size for rasterization. Set to None to allow for rasterization with the input resolution on the fly. Defaults to 224.
            use_norm (bool, optional): Whether to use norm layers in FeatUp. Refer to https://github.com/mhamilton723/FeatUp?tab=readme-ov-file#using-pretrained-upsamplers. Defaults to True.
            feat_backbone (str, optional): Feature backbone for FeatUp. Select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]. Defaults to "dino16".
            featup_weights (str | Path, optional): Weight path for FeatUp upsampler. Defaults to "mhamilton723/FeatUp".
            dust3r_weights (str | Path, optional): Weight path for DUSt3R. Defaults to "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt".
        """
        super().__init__()
        self.img_size = img_size
        self.upsampler = torch.hub.load(featup_weights, feat_backbone, use_norm=use_norm)
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(dust3r_weights)

        raster_settings = PointsRasterizationSettings(
            image_size=img_size, 
            points_per_pixel=10,
            bin_size=0,
            **kwargs
        )

        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)
        self.compositor = AlphaCompositor()


        freeze(self.dust3r)
        freeze(self.upsampler)

    def render(
        self, 
        point_clouds: Pointclouds, 
        **kwargs
    ) -> Tuple[
            Float[Tensor, "b h w c"], 
            Float[Tensor, "b 2 h w n"]
        ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render 

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf

    def forward(
        self, 
        images: Float[Tensor, "b 2 c h w"], 
        return_score_map: bool=False, 
        return_projections: bool=False
    ) -> Tuple[
            float, 
            Bool[Tensor, "b h w"], 
            Float[Tensor, "b h w"] | None, 
            Float[Tensor, "b 2 c h w"] | None
        ]:
        
        """Forward function to compute MET3R
        Args:
            images (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
            return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted), 
            return_projections (bool, False): Return projected feature maps

        Return:
            score (Float[Tensor, "b"]): MET3R score which consists of weighted mean of feature dissimlarity
            mask (bool[Tensor, "b c h w"])
        """
        
        *_, h, w = images.shape
        
        # Set rasterization settings on the fly based on input resolution
        if self.img_size is None:
            raster_settings = PointsRasterizationSettings(
                    image_size=(h, w), 
                    radius = 0.01,
                    points_per_pixel = 10,
                    bin_size=0
                )
            self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        # K=2 since we only compare an image pairs
        # NOTE: Apply DUST3R to get point maps and confidence scores
        view1 = {"img": images[:, 0, ...], "instance": [""]}
        view2 = {"img": images[:, 1, ...], "instance": [""]}
        pred1, pred2 = self.dust3r(view1, view2)

        ptmps = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1).detach()
        conf = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()

        # NOTE: Get canonical point map using the confidences
        confs11 = conf.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmps).sum(1) / confs11.sum(1)
        
        # Define principal point
        pp = torch.tensor([w /2 , h / 2], device=canon.device)
        
        
        # NOTE: Estimating fx and fy for a given canonical point map
        B, H, W, THREE = canon.shape
        assert THREE == 3

        # centered pixel grid
        pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
        canon = canon.flatten(1, 2)  # (B, HW, 3)

        # direct estimation of focal
        u, v = pixels.unbind(dim=-1)
        x, y, z = canon.unbind(dim=-1)
        fx_votes = (u * z) / x
        fy_votes = (v * z) / y

        # assume square pixels, hence same focal for X and Y
        f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
        focal = torch.nanmedian(f_votes, dim=-2)[0]
        
        # Normalized focal length
        focal[..., 0] = 1 + focal[..., 0]/w
        focal[..., 1] = 1 + focal[..., 1]/h
        focal = repeat(focal, "b c -> (b k) c", k=2)
        
        # NOTE: Getting high-resolution features from FeatUp
        b, k, *_ = images.shape
        images = rearrange(images, "b k c h w -> (b k) c h w")
        images = (images + 1) / 2
        hr_feat = self.upsampler(norm(images))

        # Some feature backbone may result in a different resolution feature maps than the inputs
        hr_feat = torch.nn.functional.interpolate(hr_feat, (images.shape[-2:]), mode="bilinear")
        hr_feat = rearrange(hr_feat, "... c h w -> ... (h w) c")
        
        # NOTE: Unproject feature on the point cloud
        features = hr_feat
        ptmps = rearrange(ptmps, "b k h w c -> (b k) (h w) c", b=b, k=2)
        point_cloud = Pointclouds(points=ptmps, features=features)
        
        # NOTE: Project and Render
        R = torch.eye(3)
        R[0, 0] *= -1
        R[1, 1] *= -1
        R = repeat(R, "... -> (b k) ...", b=b, k=2)
        T = torch.zeros((3, ))
        T = repeat(T, "... -> (b k) ...", b=b, k=2)

        # Define Pytorch3D camera for projection
        cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)
        
        # Render via point rasterizer to get projected features
        with torch.autocast("cuda", enabled=False):
            rendering, zbuf = self.render(point_cloud, cameras=cameras, background_color=[-10000] * features.shape[-1])
        rendering = rearrange(rendering, "(b k) ... -> b k ...",  b=b, k=2)
        proj_feats = rendering

        # Compute overlapping mask
        non_overlap_mask = (rendering == -10000)
        overlap_mask = (1 - non_overlap_mask.float()).prod(-1).prod(1)


        # Mask for weighted sum
        mask = overlap_mask
        
        # NOTE: Uncomment for incorporating occlusion masks along with overlap mask
        # zbuf = rearrange(zbuf, "(b k) ... -> b k ...",  b=b, k=2)
        # closest_z = zbuf[..., 0]
        # diff = (closest_z[:, 0, ...] - closest_z[:, 1, ...]).abs()
        # mask = (~(diff > 0.5) * (closest_z != -1).prod(1)) * mask

        # Get feature dissimilarity score map
        feat_dissim_maps = 1 - (proj_feats[:, 1, ...] * proj_feats[:, 0, ...]).sum(-1) / (torch.linalg.norm(proj_feats[:, 1, ...], dim=-1) * torch.linalg.norm(proj_feats[:, 0, ...], dim=-1) + 1e-3)  
            
        # Weight feature dissimilarity score map with computed mask
        feat_dissim_weighted = (feat_dissim_maps * mask).sum(-1).sum(-1)  / (mask.sum(-1).sum(-1) + 1e-6)
        
        outputs = (feat_dissim_weighted, mask)

        if return_score_map:
            outputs += feat_dissim_maps
        
        if return_projections:
            outputs += proj_feats


        return outputs

