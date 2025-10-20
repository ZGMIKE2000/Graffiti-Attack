import os 
import numpy as np
import cv2
from typing import List, Tuple
def yolo_bbox_to_pixel(yolo_bbox, img_w, img_h):
    """
    Converts YOLO bbox [cx, cy, w, h] (normalized) to [x_min, y_min, x_max, y_max] (pixels)
    """
    cx, cy, w, h = yolo_bbox
    bbox_w = int(w * img_w)
    bbox_h = int(h * img_h)
    x_min = int((cx - w / 2) * img_w)
    y_min = int((cy - h / 2) * img_h)
    x_max = x_min + bbox_w
    y_max = y_min + bbox_h
    # Ensure bbox is within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    return [x_min, y_min, x_max, y_max]

def validate_bbox_yolo(bbox_yolo):
    if not isinstance(bbox_yolo, (list, tuple)) or len(bbox_yolo) != 4:
        raise ValueError(f"YOLO bbox must be a list or tuple of 4 floats, got: {bbox_yolo}")
    for v in bbox_yolo:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"YOLO bbox values must be in [0, 1], got: {bbox_yolo}")

def get_bbox_for_image(image_path, bbox_folder=None, target_class_id=22):
    """
    Reads YOLO bbox from a .txt file corresponding to the image and filters for a specific class ID.

    Args:
        image_path (str): Path to the image file.
        bbox_folder (str, optional): Folder containing the YOLO label files.
        target_class_id (int): The target class ID to filter for.

    Returns:
        List[List[float]]: List of YOLO bounding boxes [x_center, y_center, width, height] for the target class.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    bbox_path = os.path.join(bbox_folder if bbox_folder else os.path.dirname(image_path), base + ".txt")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"BBox file not found for {image_path} (expected {bbox_path})")
    

    bbox_yolo = None
    with open(bbox_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == target_class_id:  # Check if the class ID matches
                bbox_yolo = list(map(float, parts[1:5]))
    
    return bbox_yolo

def get_classes_for_image(image_path, bbox_folder=None):
    """
    Reads all class IDs from the YOLO label file for the image.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    bbox_path = os.path.join(bbox_folder if bbox_folder else os.path.dirname(image_path), base + ".txt")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"BBox file not found for {image_path} (expected {bbox_path})")
    class_ids = []
    with open(bbox_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_ids.append(int(parts[0]))
    return class_ids

def evaluate_patch_effectiveness_single(
    yolov8_model, patched_image_np: np.ndarray, target_class_id: int
) -> float:
    """
    Evaluate the effectiveness of a patch for a single target class.

    Args:
        yolov8_model: YOLOv8 model for evaluation.
        patched_image_np (np.ndarray): Patched image.
        target_class_id (int): Target class ID.

    Returns:
        float: Highest confidence for the target class (lower is better).
    """
    results = yolov8_model(patched_image_np, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 0.0  # No detections = perfect fooling

    res = results[0].boxes
    return max(
        (float(res.conf[i]) for i, _ in enumerate(res.xyxy) if int(res.cls[i]) == target_class_id),
        default=0.0,
    )