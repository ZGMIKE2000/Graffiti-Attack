import numpy as np
import random

def compute_patch_position(
    bbox,
    patch_scale=0.5,
    position_mode='center'
):
    """
    Given a bbox (x_min, y_min, x_max, y_max), patch_scale, and position_mode,
    returns (patch_x_min, patch_y_min, patch_width, patch_height)
    """
    def _to_scalar(val):
        arr = np.array(val)
        if arr.size == 1:
            return int(arr.item())
        else:
            return int(arr.flat[0])
    x_min, y_min, x_max, y_max = [_to_scalar(val) for val in bbox[:4]]
    target_width = x_max - x_min
    target_height = y_max - y_min
    patch_width = min(int(target_width * patch_scale), target_width)
    patch_height = min(int(target_height * patch_scale), target_height)
    if position_mode == 'center':
        patch_x_min = x_min + (target_width - patch_width) // 2
        patch_y_min = y_min + (target_height - patch_height) // 2
    elif position_mode == 'random':
        patch_x_min = x_min + random.randint(0, target_width - patch_width)
        patch_y_min = y_min + random.randint(0, target_height - patch_height)
    else:
        raise ValueError("position_mode must be 'center' or 'random'")
    return patch_x_min, patch_y_min, patch_width, patch_height