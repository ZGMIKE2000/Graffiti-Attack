import os
import logging
import cv2
import torch
from utils import yolo_bbox_to_pixel
from optimization import run_es_optimization

def build_dataset(images_path, bbox_folder=None, target_class_id=None):
    """
    Build a dataset of images and their bounding boxes for the target class.

    Args:
        images_path (str or List[str]): Path to a single image or a list of image paths.
        bbox_folder (str): Folder containing YOLO bbox .txt files (batch mode).
        target_class_id (int): Target class ID to filter bounding boxes.
        bbox_path (str): Path to a single YOLO bbox .txt file (single image mode).

    Returns:
        List[Tuple[np.ndarray, List[Tuple[int, int, int, int]], str]]:
            A list of tuples containing the image, bounding boxes, and image path.
    """
    real_sign_images_dataset = []

    # Handle single image mode
    if isinstance(images_path, str):
        logging.info("Single file mode detected. Building dataset for one image...")
        img_np = cv2.imread(images_path)
        if img_np is None:
            logging.error(f"Image could not be read: {images_path}")
            raise ValueError(f"Invalid image path: {images_path}")

        img_h, img_w = img_np.shape[:2]
        bboxes = []
        with open(bbox_folder, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                bbox_yolo = list(map(float, parts[1:5]))
                if cls_id == target_class_id:
                    x1, y1, x2, y2 = yolo_bbox_to_pixel(bbox_yolo, img_w, img_h)
                    bboxes.append((x1, y1, x2, y2))

        if not bboxes:
            logging.error(f"No bounding boxes found for target class {target_class_id} in {bbox_folder}")
            raise ValueError(f"No bounding boxes found in {bbox_folder}")

        real_sign_images_dataset.append((img_np, bboxes, images_path))

    # Handle batch mode
    elif isinstance(images_path, list) and bbox_folder:
        logging.info("Batch mode detected. Building dataset...")
        for image_path in images_path:
            img_np = cv2.imread(image_path)
            if img_np is None:
                logging.warning(f"Skipping {image_path}: image could not be read")
                continue
            img_h, img_w = img_np.shape[:2]
            bbox_file = os.path.join(bbox_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            bboxes = []
            with open(bbox_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    bbox_yolo = list(map(float, parts[1:5]))
                    if cls_id == target_class_id:
                        x1, y1, x2, y2 = yolo_bbox_to_pixel(bbox_yolo, img_w, img_h)
                        bboxes.append((x1, y1, x2, y2))
            if bboxes:
                real_sign_images_dataset.append((img_np, bboxes, image_path))

    else:
        raise ValueError("Invalid input: 'images_path' must be a string or list, and 'bbox_folder' or 'bbox_path' must be provided.")

    logging.info(f"Loaded {len(real_sign_images_dataset)} images for optimization.")
    return real_sign_images_dataset

def process_image(
    G_frozen, yolo_model, images_path, bbox_folder, target_class_id,
    num_generations, population_size, output_dir, checkpoint_dir, bbox_path=None
):
    """
    Process images for black-box patch optimization.

    Args:
        G_frozen: The generator model.
        yolo_model: The YOLO model.
        images_path: Path to a single image or a list of image paths.
        bbox_folder: Folder containing YOLO bbox .txt files (batch mode).
        target_class_id: Target class ID for optimization.
        num_generations: Number of generations for optimization.
        population_size: Population size for optimization.
        output_dir: Directory to save results.
        checkpoint_dir: Directory to save checkpoints.
        bbox_path: Path to a single YOLO bbox .txt file (single image mode).
    """

    # Debugging logs
    logging.info(f"images_path: {images_path}")
    logging.info(f"bbox_folder: {bbox_folder}")
    logging.info(f"bbox_path: {bbox_path}")
    # Build the dataset
    img_dataset = build_dataset(images_path, bbox_folder, target_class_id)
    # If yolo_model is a list, pass it as ensemble to optimization
    if isinstance(yolo_model, list):
        logging.info(f"Using YOLO ensemble with {len(yolo_model)} models.")
    else:
        logging.info("Using single YOLO model.")

    run_es_optimization(
        G_frozen, yolo_model, img_dataset,
        target_class_id,
        num_generations=num_generations, population_size=population_size,
        checkpoint_dir=checkpoint_dir
    )