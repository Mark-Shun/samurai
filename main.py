import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys

# Add custom library path to the system path
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor # Import Sam2's video predictor

# Global variables
color = [(255, 0, 0)]
result_prefix = "Alpha-"
prompts = {}

# Load bounding boxes prompts from text file
# TODO: Handle multiple Bboxes or Keyframes
def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        if fid not in prompts:
            prompts[fid] = []
        prompts[fid].append(((x, y, x + w, y + h), 0)) # Store bounding box and label
    return prompts

# Determine appropiate model configuration
def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

# Validate and prepare video frames or frame directory path
def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    if args.save_to_video or args.save_to_img_seq:
        if osp.isdir(args.video_path): # If input is a directory of frames
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
            fps=30
        else: # If input is a video file
            cap = cv2.VideoCapture(args.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")
        # Setup video writer    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, fps, (width, height))
    
    # Prepare output folder for image sequence
    if args.save_to_img_seq:
        video_path = args.video_path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        root_folder = os.path.dirname(video_path)
        if not os.path.exists(os.path.join(root_folder, result_prefix + video_name)):
            os.mkdir(os.path.join(root_folder, result_prefix + video_name))
        output_path = os.path.join(root_folder, result_prefix + video_name)

    # Use the model for tracking and segmentation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        
        # Initialize objects for tracking using all prompts from the first frame
        # TODO: Handle multiple Bboxes or Keyframes
        for bbox, track_label in prompts[0]:  # Initialize each prompt
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=track_label)

        # Process frames and save results
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}  # Dictionary to hold masks for visualization
            bbox_to_vis = {}  # Dictionary to hold bounding boxes for visualization

            # Extract bounding boxes and masks for each object
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
            
            if args.save_to_img_seq: # Save mask frames as png sequence
                for obj_id, mask in mask_to_vis.items():
                    img = loaded_frames[frame_idx].copy()
                    mask_img = np.zeros((height, width, 4), np.uint8)
                    mask_img[mask] = (255,255,255,255)                    
                    cv2.imwrite(os.path.join(output_path, (f"{result_prefix}{video_name}_{obj_id:03d}_{frame_idx:08d}_.png")), mask_img)

            if args.save_to_video: # Overlay masks and bounding boxes on video frames
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                if args.show_bbox: # Optionally show the bbox
                    for obj_id, bbox in bbox_to_vis.items():
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()

    # Clean up resources
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=False, help="Save results to a video.")
    parser.add_argument("--save_to_img_seq", default=True, help="Save results to a sequence of png images.")
    parser.add_argument("--show_mask", default=False, help="Show the mask.")
    parser.add_argument("--show_bbox", default=False, help="Show the bbox.")
    args = parser.parse_args()
    main(args)