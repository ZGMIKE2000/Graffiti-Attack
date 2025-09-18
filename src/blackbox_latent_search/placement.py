import numpy as np
import random

def compute_patch_position(
    bbox,
    patch_scale,
    position_mode='center',
    x_offset=0.0,  # Used only in 'free' mode
    y_offset=0.0   # Used only in 'free' mode
):
    """
    Given a bbox (x_min, y_min, x_max, y_max), patch_scale, position_mode, and optional offsets,
    returns (patch_x_min, patch_y_min, patch_width, patch_height).

    Args:
        bbox (tuple): Bounding box (x_min, y_min, x_max, y_max).
        patch_scale (float): Scale of the patch relative to the bounding box.
        position_mode (str): Placement mode ('center', 'random', or 'free').
        x_offset (float): Horizontal offset as a fraction of bbox width (used in 'free' mode).
        y_offset (float): Vertical offset as a fraction of bbox height (used in 'free' mode').

    Returns:
        tuple: (patch_x_min, patch_y_min, patch_width, patch_height).
    """
    def _to_scalar(val):
        arr = np.array(val)
        if arr.size == 1:
            return int(arr.item())
        else:
            return int(arr.flat[0])

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = [_to_scalar(val) for val in bbox[:4]]
    target_width = x_max - x_min
    target_height = y_max - y_min

    # Compute patch dimensions
    patch_width = min(int(target_width * patch_scale), target_width)
    patch_height = min(int(target_height * patch_scale), target_height)

    # Compute patch position based on the mode
    if position_mode == 'center':
        patch_x_min = x_min + (target_width - patch_width) // 2
        patch_y_min = y_min + (target_height - patch_height) // 2
    elif position_mode == 'random':
        patch_x_min = x_min + random.randint(0, target_width - patch_width)
        patch_y_min = y_min + random.randint(0, target_height - patch_height)
    elif position_mode == 'free':
        # Apply offsets in 'free' mode
        patch_x_min = x_min + int(x_offset * target_width)
        patch_y_min = y_min + int(y_offset * target_height)
    else:
        raise ValueError("position_mode must be 'center', 'random', or 'free'")

    # Ensure the patch stays within the bounding box
    patch_x_min = max(x_min, min(patch_x_min, x_max - patch_width))
    patch_y_min = max(y_min, min(patch_y_min, y_max - patch_height))

    return patch_x_min, patch_y_min, patch_width, patch_height