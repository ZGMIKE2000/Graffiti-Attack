import json
import os
import cv2
import numpy as np

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def load_image(path, resize=None):
    img = cv2.imread(path)
    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
    return img

def bbox_from_yolo_txt_file(img_path, label_path_or_dir, target_class=None):
    """
    Read YOLO .txt for the image and return list of (cls, x1, y1, x2, y2) in pixel coords.
    label_path_or_dir: either a full .txt path or the directory that contains the .txt
    If target_class is given, returns only boxes of that class.
    """
    import os
    from PIL import Image

    base = os.path.splitext(os.path.basename(img_path))[0]
    # accept either file path or directory
    if os.path.isdir(label_path_or_dir):
        txtp = os.path.join(label_path_or_dir, base + ".txt")
    else:
        # if a full path to a .txt was passed, use it; otherwise try same-dir lookup
        if label_path_or_dir.endswith(".txt"):
            txtp = label_path_or_dir
        else:
            txtp = os.path.join(os.path.dirname(label_path_or_dir), base + ".txt")

    if not os.path.exists(txtp):
        return []

    img = Image.open(img_path)
    orig_w, orig_h = img.size

    bboxes = []
    with open(txtp, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = map(float, parts[:5])
            # YOLO normalized cx,cy,w,h -> pixel
            xc *= orig_w; yc *= orig_h; w *= orig_w; h *= orig_h
            x1 = int(round(xc - w/2.0)); y1 = int(round(yc - h/2.0))
            x2 = int(round(xc + w/2.0)); y2 = int(round(yc + h/2.0))
            if target_class is not None and int(cls) != int(target_class):
                continue
            # clamp
            x1 = max(0, min(orig_w-1, x1))
            x2 = max(1, min(orig_w, x2))
            y1 = max(0, min(orig_h-1, y1))
            y2 = max(1, min(orig_h, y2))
            bboxes.append((int(cls), x1, y1, x2, y2))
    return bboxes