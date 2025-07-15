import numpy as np
import cv2
import random


def extract_graffiti_mask_from_patch(patch_np, border=10, quantile=75):
    """
    Improved: Robustly estimate background from clean border pixels.
    """
    h, w = patch_np.shape[:2]
    
    # Select center slices of each border (avoids corners)
    border_pixels = np.concatenate([
        patch_np[:border, :, :3].reshape(-1, 3),
        patch_np[-border:, :, :3].reshape(-1, 3),
        patch_np[:, :border, :3].reshape(-1, 3),
        patch_np[:, -border:, :3].reshape(-1, 3)
    ])
    
    # Remove outliers
    med = np.median(border_pixels, axis=0)
    dists = np.linalg.norm(border_pixels - med, axis=1)
    clean_pixels = border_pixels[dists < np.percentile(dists, 80)]
    bg_color = np.median(clean_pixels, axis=0)

    # Color distance for whole image
    color_dist = np.linalg.norm(patch_np[:, :, :3].astype(np.float32) - bg_color, axis=2)

    # Dynamic threshold: based on image contrast
    threshold = np.percentile(color_dist, quantile)  # e.g. top 5% as graffiti
    graffiti_mask = (color_dist > threshold).astype(np.uint8) * 255

    # Cleanup
    graffiti_mask = cv2.medianBlur(graffiti_mask, 5)
    graffiti_mask = cv2.threshold(graffiti_mask, 127, 255, cv2.THRESH_BINARY)[1]
    graffiti_mask = cv2.dilate(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)
    graffiti_mask = cv2.erode(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)

    return graffiti_mask


def extract_graffiti_rgba_from_patch(patch_np, graffiti_mask):
    """
    Returns the graffiti patch with transparent background as an RGBA image.
    graffiti_mask: uint8 mask, 255=graffiti, 0=background
    """
    # Ensure mask is binary and single channel
    alpha = graffiti_mask.astype(np.uint8)
    # If patch is RGB, add alpha channel
    if patch_np.shape[2] == 3:
        rgba = np.dstack([patch_np, alpha])
    else:
        rgba = patch_np.copy()
        rgba[:, :, 3] = alpha
    # Optional: set background pixels to 0 (transparent)
    rgba[alpha == 0] = [0, 0, 0, 0]
    return rgba


def realistic_patch_applier(
    graffiti_rgba_patch_np,  # RGBA graffiti patch (H, W, 4)
    sign_image_np,
    target_bbox_on_sign,
    alpha=0.7,
    patch_scale=0.5,
    position_mode='center'  # 'center' or 'random'
):
    """
    Applies the RGBA patch to a smaller area inside the bbox.
    patch_scale: float in (0, 1], fraction of bbox to cover (e.g., 0.5 = 50%)
    position_mode: 'center' or 'random'
    """

    if not (isinstance(target_bbox_on_sign, (list, tuple, np.ndarray)) and len(target_bbox_on_sign) == 4):
        raise ValueError(f"Invalid bbox: {type(target_bbox_on_sign)}, shape: {getattr(target_bbox_on_sign, 'shape', None)}, value: {target_bbox_on_sign}")
    sign_image_float = sign_image_np.astype(np.float32)
    patched_image_np = sign_image_float.copy()
    # print("DEBUG target_bbox_on_sign:", target_bbox_on_sign, "len:", len(target_bbox_on_sign))
    def _to_scalar(val):
        arr = np.array(val)
        if arr.size == 1:
            return int(arr.item())
        else:
            # If it's an array with more than one element, take the first element
            return int(arr.flat[0])

    x_min, y_min, x_max, y_max = [_to_scalar(val) for val in target_bbox_on_sign[:4]]    
    
    target_width = x_max - x_min
    target_height = y_max - y_min

    # Scale patch size
    patch_width = int(target_width * patch_scale)
    patch_height = int(target_height * patch_scale)

    # Ensure patch fits in bbox
    patch_width = min(patch_width, target_width)
    patch_height = min(patch_height, target_height)

    if position_mode == 'center':
        patch_x_min = x_min + (target_width - patch_width) // 2
        patch_y_min = y_min + (target_height - patch_height) // 2
    elif position_mode == 'random':
        patch_x_min = x_min + random.randint(0, target_width - patch_width)
        patch_y_min = y_min + random.randint(0, target_height - patch_height)
    else:
        raise ValueError("position_mode must be 'center' or 'random'")

    patch_x_max = patch_x_min + patch_width
    patch_y_max = patch_y_min + patch_height

    # Resize RGBA graffiti patch
    graffiti_resized = cv2.resize(graffiti_rgba_patch_np, (patch_width, patch_height), interpolation=cv2.INTER_AREA)
    patch_rgb = graffiti_resized[:, :, :3].astype(np.float32)
    patch_alpha = (graffiti_resized[:, :, 3].astype(np.float32) / 255.0)

    roi = patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3]
    for c in range(3):
        roi[:, :, c] = patch_alpha * (alpha * patch_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]) + (1 - patch_alpha) * roi[:, :, c]

    patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3] = roi
    return patched_image_np.astype(np.uint8)