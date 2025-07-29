import torch
from met3r import MEt3R

IMG_SIZE = 224

# Initialize MEt3R
metric = MEt3R(
    img_size=IMG_SIZE,
    use_norm=True,
    # backbone="vggt",
    backbone="mast3r",
    feature_backbone="dino16",
    feature_backbone_weights="mhamilton723/FeatUp",
    upsampler="featup",
    distance="cosine",
    freeze=True, 
)
metric = metric.cuda()

# Prepare inputs of shape (batch, views, channels, height, width): views must be 2
# RGB range must be in [-1, 1]
# Reduce the batch size in case of CUDA OOM
inputs = torch.randn((1, 2, 3, IMG_SIZE, IMG_SIZE)).cuda()
inputs = inputs.clip(-1, 1)

image_names = ["/data1/usman/vision/met3r/vggt/resized_image2.png", "/data1/usman/vision/met3r/vggt/resized_image2.png"]
# images = load_and_preprocess_images(image_names)
# Evaluate MEt3R
score, *_ = metric(
    images=inputs, 
    image_names=image_names,
    return_overlap_mask=False, # Default 
    return_score_map=False, # Default 
    return_projections=False # Default 
)

# Should be between 0.17 - 0.18
print(score.mean().item())

# Clear up GPU memory
torch.cuda.empty_cache()
