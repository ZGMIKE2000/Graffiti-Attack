import numpy as np
import cv2


def extract_graffiti_mask_from_patch(patch_np, border=10, threshold=73):
    """
    Extracts a binary graffiti mask from a patch using color distance from the border.
    Args:
        patch_np (np.array): (H, W, 3) uint8 image.
        border (int): Border width (pixels) to estimate background color.
        threshold (float): Color distance threshold for graffiti.
    Returns:
        np.array: Binary mask (H, W) uint8, 255=graffiti, 0=background.
    """
    h, w = patch_np.shape[:2]
    border_pixels = np.concatenate([
        patch_np[:border, :, :3].reshape(-1, 3),
        patch_np[-border:, :, :3].reshape(-1, 3),
        patch_np[:, :border, :3].reshape(-1, 3),
        patch_np[:, -border:, :3].reshape(-1, 3)
    ])
    bg_color = np.median(border_pixels, axis=0)
    color_dist = np.linalg.norm(patch_np[:, :, :3].astype(np.float32) - bg_color, axis=2)
    graffiti_mask = (color_dist > threshold).astype(np.uint8) * 255
    graffiti_mask = cv2.medianBlur(graffiti_mask, 5)
    graffiti_mask = cv2.threshold(graffiti_mask, 127, 255, cv2.THRESH_BINARY)[1]
    graffiti_mask = cv2.dilate(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)
    graffiti_mask = cv2.erode(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)
    return graffiti_mask


def realistic_patch_applier(graffiti_patch_np, graffiti_mask_np, sign_image_np, target_bbox_on_sign, alpha=0.7, patch_scale=0.5):
    """
    Applies the patch to a smaller area inside the bbox, centered.
    patch_scale: float in (0, 1], fraction of bbox to cover (e.g., 0.5 = 50%)
    """
    sign_image_float = sign_image_np.astype(np.float32)
    patched_image_np = sign_image_float.copy()

    x_min, y_min, x_max, y_max = [int(val) for val in target_bbox_on_sign]
    target_width = x_max - x_min
    target_height = y_max - y_min

    # Scale patch size
    patch_width = int(target_width * patch_scale)
    patch_height = int(target_height * patch_scale)

    # Center the patch in the bbox
    patch_x_min = x_min + (target_width - patch_width) // 2
    patch_y_min = y_min + (target_height - patch_height) // 2
    patch_x_max = patch_x_min + patch_width
    patch_y_max = patch_y_min + patch_height

    graffiti_resized = cv2.resize(graffiti_patch_np, (patch_width, patch_height), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(graffiti_mask_np, (patch_width, patch_height), interpolation=cv2.INTER_NEAREST)
    patch_rgb = graffiti_resized.astype(np.float32)
    patch_alpha = (mask_resized.astype(np.float32) / 255.0)

    roi = patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3]
    for c in range(3):
        roi[:, :, c] = patch_alpha * (alpha * patch_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]) + (1 - patch_alpha) * roi[:, :, c]

    patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3] = roi
    return patched_image_np.astype(np.uint8)
