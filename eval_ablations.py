from met3r import MEt3R
import os
import torch
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm


def eval(args):
    metric = MEt3R(
        img_size=args.img_size,
        use_norm=True,
        backbone=args.backbone,
        feature_backbone=args.feature_backbone,
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance="cosine",
        freeze=True, 
    ).cuda()

    frame_scores = []
    if args.experiment == "random":
        print("Running random ablation")

        for _ in tqdm(range(10)):
            inputs = torch.randn((10, 2, 3, args.img_size, args.img_size)).cuda()
            inputs = inputs.clip(-1, 1)
            score, *_ = metric(
                images=inputs,
                return_overlap_mask=False,  # Default
                return_score_map=False,  # Default
                return_projections=False  # Default
            )

            frame_scores.extend(score.cpu().numpy().tolist())
        print("Mean score for random ablation:", sum(frame_scores) / len(frame_scores))

    elif args.experiment == "real":
        print("Running real video ablation")

        transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # to [-1, 1]
    ])

        # Load and sort image files
        image_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".png")
        ])

        # Load and transform all images
        images = [transform(Image.open(f).convert("RGB")) for f in image_files]

        # Make sliding window pairs: (i, i+1)
        pairs = [torch.stack([images[i], images[i + 1]]) for i in range(len(images) - 1)]  # (2, 3, H, W) each

        # Group pairs in batches of 10
        batch_size = 10
        total_pairs = len(pairs)
        num_batches = (total_pairs + batch_size - 1) // batch_size  # ceil division

        # Iterate over batches
        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_pairs)
            batch_pairs = pairs[start:end]  # up to 10 pairs, may be fewer in the last batch
            inputs = torch.stack(batch_pairs).cuda()  # Shape: (B, 2, 3, H, W)
            inputs = inputs.clamp(-1, 1)

            # Run the metric
            score, *_ = metric(
                images=inputs,
                return_overlap_mask=False,  # Default
                return_score_map=False,  # Default
                return_projections=False  # Default
            )
            
            frame_scores.extend(score.cpu().numpy().tolist())

        print("Mean score for real video ablation:", sum(frame_scores) / len(frame_scores))

    # save the scores to a file
    output_file = "experiments.json"
    experiment = {
        "experiment": args.experiment,
        "backbone": args.backbone,
        "feature_backbone": args.feature_backbone,
        "scores": frame_scores
    }
    with open(output_file, "a") as f:
        f.write(json.dumps(experiment) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--backbone", type=str, default="dust3r", help="Backbone model to use.")
    parser.add_argument("--input_dir", type=str, default="/data1/usman/vision/data/output_frames", help="Directory containing input data.")
    parser.add_argument("--feature_backbone", type=str, default="dino16", help="Feature backbone model to use.")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for the model.")
    parser.add_argument("--experiment", type=str, choices=["random", "real"], default="random", help="Type of experiment to run.")

    args = parser.parse_args()
    eval(args)