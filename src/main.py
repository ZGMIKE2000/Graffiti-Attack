import os
import sys
import torch
import click
import logging
import yaml
import random
import numpy as np

sys.path.append('/home/michele/hdd/stylegan3_error')

from processing import process_image
from models import load_stylegan3_generator, load_yolov8_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_config(cli_args, yaml_config):
    """Merge CLI arguments with YAML config, giving priority to CLI."""
    return {key: cli_args.get(key) or yaml_config.get(key) for key in yaml_config.keys()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@click.command()
@click.option('--config', type=click.Path(exists=True), help="YAML config file with arguments")
@click.option('--gan-snapshot-path', required=True, type=click.Path(exists=True), help="Path to StyleGAN3 .pkl")
@click.option('--yolo-model-path', required=True, type=click.Path(exists=True), help="Path to YOLOv8 .pt")
@click.option('--img-path', type=click.Path(), help="Path to a single sign image")
@click.option('--img-folder', type=click.Path(), help="Folder with sign images (batch mode)")
@click.option('--bbox-path', type=click.Path(exists=True), default=None, help="Path to YOLO bbox .txt file (single image mode)")
@click.option('--bbox-folder', type=click.Path(), help="Folder with YOLO bbox .txt files (batch mode, default: same as image folder)")
@click.option('--target-class-id', required=True, type=int, help="Class ID to fool")
@click.option('--target-misclassify-id', type=int, default=None, help="Target class ID for misclassification (optional)")
@click.option('--num-generations', type=int, default=100, help="Number of generations for optimization")
@click.option('--population-size', type=int, default=10, help="Population size for optimization")
# @click.option('--alpha', type=float, default=1.0, help="alpha value for optimization")
# @click.option('--patch-scale', type=float, default=1.0, help="patch scale for optimization")
@click.option('--checkpoint-dir', type=click.Path(), default="checkpoints", show_default=True, help="Directory to save checkpoints")
@click.option('--output-dir', type=click.Path(), default="output", show_default=True, help="Directory to save results")
def main(gan_snapshot_path, yolo_model_path, img_path, img_folder, bbox_path, bbox_folder,
         target_class_id, target_misclassify_id, num_generations, population_size, checkpoint_dir,output_dir, config):

    # --- Load YAML config if provided ---
    if config:
        cfg = load_config(config)
        cli_args = locals()
        merged_config = merge_config(cli_args, cfg)
        locals().update(merged_config)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Set Seed for Reproducibility ---
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Using seed: {seed}")

    # --- Load Models ---
    G_frozen = load_stylegan3_generator(os.path.expanduser(gan_snapshot_path), device)
    G_frozen.eval()
    yolov8_model = load_yolov8_model(os.path.expanduser(yolo_model_path), device)

    # --- Batch or Single Image ---
    if img_folder:
        # Batch mode: process all images in the folder
        image_files = [
            os.path.join(img_folder, f) for f in os.listdir(img_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg")) and
            os.path.exists(os.path.join(bbox_folder or img_folder, os.path.splitext(f)[0] + ".txt"))
        ]

        if not image_files:
            logging.warning("No valid images with corresponding labels found in the folder.")
            sys.exit(1)

        process_image(
            G_frozen, yolov8_model,
            image_files, bbox_folder, target_class_id,
            num_generations, population_size, output_dir, checkpoint_dir
        )

    elif img_path:
        # Single image mode
        if bbox_path is None:
            logging.error("Please provide --bbox-path for single image mode.")
            sys.exit(1)

        process_image(
            G_frozen, yolov8_model,
            img_path, bbox_path, target_class_id,
            num_generations, population_size, output_dir,checkpoint_dir
        )
    else:
        # No valid input provided
        logging.error("Please provide either --img-path or --img-folder.")
        sys.exit(1)

if __name__ == "__main__":
    main()