
import argparse
import os
import torch
from met3r import MEt3R
import cv2
from torchvision import transforms
from glob import glob
from tqdm import tqdm


def load_and_preprocess_images(data_dir, img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),                     
        transforms.Resize((img_size, img_size)),   
        transforms.Normalize([0.5]*3, [0.5]*3)      
    ])

    def video_frame_generator(video_paths):
        caps = [cv2.VideoCapture(vp) for vp in video_paths]

        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    for c in caps:
                        c.release()
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(frame_rgb)
                frames.append(tensor)

            # Create sliding window pairs: (1,2), (2,3), ...
            pairs = [(frames[i], frames[i+1]) for i in range(len(frames) - 1)]
            stacked_pairs = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)  
            yield stacked_pairs  # shape: (N-1, 2, 3, img_size, img_size)

    def all_subdir_generators():
        subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

        for subdir in subdirs:
            video_paths = sorted(glob(os.path.join(subdir, '*.mp4')))
            if len(video_paths) < 2:
                print(f"Skipping {subdir}: Need at least 2 videos, found {len(video_paths)}")
                continue
            yield video_frame_generator(video_paths)

    return all_subdir_generators()
    


def evaluate_met3r(args):
    metric = MEt3R(
        img_size=args.img_size,
        use_norm=args.use_norm,
        backbone=args.backbone,
        feature_backbone=args.feature_backbone,
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance="cosine",
        freeze=True, 
    ).cuda()

    # First, get the list of valid subdirectories
    subdirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))]

    # Then call the generator loader
    generators = load_and_preprocess_images(args.data_dir, args.img_size)

    # Now wrap with tqdm
    subdir_scores = []
    for subdir_path, subdir_gen in tqdm(zip(subdirs, generators), total=len(subdirs), desc="Subdirectories"):
        frame_scores = []
        for frame_tensor in subdir_gen:
            frame_tensor = frame_tensor.cuda()
            score, *_ = metric(
                images=frame_tensor,
                return_overlap_mask=False,  # Default
                return_score_map=False,  # Default
                return_projections=False  # Default
            )
            frame_scores.append(score.mean().item())
        subdir_scores.append(sum(frame_scores)/len(frame_scores))

    average_score = sum(subdir_scores) / len(subdir_scores)
    print(f"Average MEt3R score across all subdirectories: {average_score:.4f}")




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


    args = parser.parse_args()

    evaluate_met3r(args)
