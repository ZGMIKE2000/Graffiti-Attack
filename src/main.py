"""
Graffiti Attack Patch Optimization Tool

Usage (single image):
    python main.py --stylegan3-snapshot-path /path/to/sg3.pkl --yolov8-model-path /path/to/yolov8.pt --sign-img-path myimage.jpg --bbox-yolo 0.5 0.5 0.2 0.2 --target-class-id 25

Usage (batch mode):
    python main.py --stylegan3-snapshot-path /path/to/sg3.pkl --yolov8-model-path /path/to/yolov8.pt --sign-img-folder ./images --bbox-folder ./bboxes --target-class-id 25

Or use a YAML config:
    python main.py --config myexperiment.yaml
"""
import os
import sys
import torch
import click
from tqdm import tqdm
import logging
import yaml


def load_config(config_path):
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
# Add StyleGAN3 repo to sys.path for dnnlib/legacy
sys.path.append('/home/michele/hdd/stylegan3')  # Adjust as needed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

from models import load_stylegan3_generator, load_yolov8_model
from utils import get_bbox_for_image, validate_bbox_yolo
from processing import process_image


@click.command()
@click.option('--config', type=click.Path(exists=True), help="YAML config file with arguments")
@click.option('--stylegan3-snapshot-path', required=True, type=click.Path(exists=True), help="Path to StyleGAN3 .pkl")
@click.option('--yolov8-model-path', required=True, type=click.Path(exists=True), help="Path to YOLOv8 .pt")
@click.option('--sign-img-path', type=click.Path(), help="Path to a single sign image")
@click.option('--sign-img-folder', type=click.Path(), help="Folder with sign images (batch mode)")
@click.option('--bbox-yolo', nargs=4, type=float, default=None, help="YOLO bbox [x_center, y_center, w, h] (normalized, single image mode)")
@click.option('--bbox-folder', type=click.Path(), help="Folder with YOLO bbox .txt files (batch mode, default: same as image folder)")
@click.option('--target-class-id', required=True, type=int, help="Class ID to fool")
@click.option('--target-misclassify-id', type=int, default=None, help="Target class ID for misclassification (optional)")
@click.option('--num-generations', type=int, default=100, help="Number of generations for optimization")
@click.option('--population-size', type=int, default=10, help="Population size for optimization")
@click.option('--output-dir', type=click.Path(), default="output", show_default=True, help="Directory to save results")
def main(stylegan3_snapshot_path, yolov8_model_path, sign_img_path, sign_img_folder, bbox_yolo, bbox_folder,
         target_class_id, target_misclassify_id, num_generations, population_size, output_dir, config):

    # --- Load YAML config if provided ---

    if config:
        cfg = load_config(config)
        # Only override if not set by CLI (CLI has priority)
        stylegan3_snapshot_path = cfg.get('stylegan3_snapshot_path', stylegan3_snapshot_path)
        yolov8_model_path = cfg.get('yolov8_model_path', yolov8_model_path)
        sign_img_path = cfg.get('sign_img_path', sign_img_path)
        sign_img_folder = cfg.get('sign_img_folder', sign_img_folder)
        bbox_yolo = cfg.get('bbox_yolo', bbox_yolo)
        bbox_folder = cfg.get('bbox_folder', bbox_folder)
        target_class_id = cfg.get('target_class_id', target_class_id)
        target_misclassify_id = cfg.get('target_misclassify_id', target_misclassify_id)
        num_generations = cfg.get('num_generations', num_generations)
        population_size = cfg.get('population_size', population_size)
        output_dir = cfg.get('output_dir', output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Models ---
    G_frozen = load_stylegan3_generator(os.path.expanduser(stylegan3_snapshot_path), device)
    G_frozen.eval()
    yolov8_model = load_yolov8_model(os.path.expanduser(yolov8_model_path), device)

    # --- Batch or Single Image ---
    if sign_img_folder:
        # Batch mode: process all images in the folder

        image_files = [f for f in os.listdir(sign_img_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        bbox_files = [f for f in os.listdir(bbox_folder or sign_img_folder) if f.lower().endswith(".txt")]
        if len(image_files) != len(bbox_files):
            logging.warning(f"Number of images ({len(image_files)}) and bbox files ({len(bbox_files)}) do not match.")

        image_files = [f for f in os.listdir(sign_img_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        failed_cases = []
        for fname in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(sign_img_folder, fname)
            try:
                bbox = get_bbox_for_image(img_path, bbox_folder)
                validate_bbox_yolo(bbox)
                process_image(
                    G_frozen, yolov8_model,
                    img_path,
                    bbox, target_class_id, target_misclassify_id,
                    num_generations, population_size
                )
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
                failed_cases.append((img_path, str(e)))
            continue

        if failed_cases:
            logging.warning(f"\n{len(failed_cases)} images failed to process:")
            for img_path, reason in failed_cases:
                logging.warning(f"  {img_path}: {reason}")

    elif sign_img_path:
        # Single image mode

        if bbox_yolo is None:
            logging.error("Please provide --bbox-yolo for single image mode.")
            sys.exit(1)
        validate_bbox_yolo(bbox_yolo)
        process_image(
            G_frozen, yolov8_model,
            sign_img_path,
            bbox_yolo, target_class_id, target_misclassify_id,
            num_generations, population_size
        )
    else:
        # No valid input provided

        print("Please provide either --sign-img-path or --sign-img-folder.")
        sys.exit(1)

if __name__ == "__main__":
    main()