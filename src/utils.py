import os 

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

def get_bbox_for_image(image_path, bbox_folder=None):
    """
    Reads YOLO bbox from a .txt file corresponding to the image.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    bbox_path = os.path.join(bbox_folder if bbox_folder else os.path.dirname(image_path), base + ".txt")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"BBox file not found for {image_path} (expected {bbox_path})")
    with open(bbox_path, "r") as f:
        # Assumes first line: class x_center y_center width height
        line = f.readline().strip().split()
        bbox_yolo = list(map(float, line[1:5]))
    return bbox_yolo