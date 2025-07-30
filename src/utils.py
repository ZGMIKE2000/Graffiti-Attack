import os
from typing import List, Tuple

def yolo_bbox_to_pixel(yolo_bbox: List[float], img_w: int, img_h: int) -> List[int]:
    """
    Converts YOLO bbox [cx, cy, w, h] (normalized) to [x_min, y_min, x_max, y_max] (pixels).

    Args:
        yolo_bbox (List[float]): Normalized YOLO bbox [cx, cy, w, h].
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        List[int]: Pixel bbox [x_min, y_min, x_max, y_max].
    """
    cx, cy, w, h = yolo_bbox
    x_min = max(0, int((cx - w / 2) * img_w))
    y_min = max(0, int((cy - h / 2) * img_h))
    x_max = min(img_w, int((cx + w / 2) * img_w))
    y_max = min(img_h, int((cy + h / 2) * img_h))
    return [x_min, y_min, x_max, y_max]

def validate_bbox_yolo(bbox_yolo: List[float]) -> None:
    """
    Validates a YOLO bbox to ensure it is a list of 4 floats in the range [0, 1].

    Args:
        bbox_yolo (List[float]): YOLO bbox [cx, cy, w, h].

    Raises:
        ValueError: If the bbox is invalid.
    """
    if not isinstance(bbox_yolo, (list, tuple)) or len(bbox_yolo) != 4:
        raise ValueError(f"YOLO bbox must be a list or tuple of 4 floats, got: {bbox_yolo}")
    if not all(0.0 <= v <= 1.0 for v in bbox_yolo):
        raise ValueError(f"YOLO bbox values must be in [0, 1], got: {bbox_yolo}")

def _read_yolo_file(image_path: str, bbox_folder: str = None) -> List[str]:
    """
    Reads the YOLO label file corresponding to an image.

    Args:
        image_path (str): Path to the image file.
        bbox_folder (str, optional): Folder containing YOLO label files. Defaults to the image's folder.

    Returns:
        List[str]: Lines from the YOLO label file.

    Raises:
        FileNotFoundError: If the label file does not exist.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    bbox_path = os.path.join(bbox_folder if bbox_folder else os.path.dirname(image_path), base + ".txt")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"BBox file not found for {image_path} (expected {bbox_path})")
    with open(bbox_path, "r") as f:
        return f.readlines()

def get_bbox_for_image(image_path: str, bbox_folder: str = None) -> List[float]:
    """
    Reads the YOLO bbox for an image from its label file.

    Args:
        image_path (str): Path to the image file.
        bbox_folder (str, optional): Folder containing YOLO label files. Defaults to the image's folder.

    Returns:
        List[float]: YOLO bbox [cx, cy, w, h].

    Raises:
        FileNotFoundError: If the label file does not exist.
        ValueError: If the bbox format is invalid.
    """
    lines = _read_yolo_file(image_path, bbox_folder)
    if not lines:
        raise ValueError(f"No bbox data found in label file for {image_path}")
    # Assumes first line: class x_center y_center width height
    bbox_yolo = list(map(float, lines[0].strip().split()[1:5]))
    validate_bbox_yolo(bbox_yolo)
    return bbox_yolo

def get_classes_for_image(image_path: str, bbox_folder: str = None) -> List[int]:
    """
    Reads all class IDs from the YOLO label file for an image.

    Args:
        image_path (str): Path to the image file.
        bbox_folder (str, optional): Folder containing YOLO label files. Defaults to the image's folder.

    Returns:
        List[int]: List of class IDs.

    Raises:
        FileNotFoundError: If the label file does not exist.
    """
    lines = _read_yolo_file(image_path, bbox_folder)
    class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()]
    return class_ids