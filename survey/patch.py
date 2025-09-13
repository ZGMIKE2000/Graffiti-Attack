import numpy as np
import cv2
from placement import compute_patch_position

def extract_graffiti_mask_from_patch(patch_np, border=10, quantile=75):
    h, w = patch_np.shape[:2]
    border_pixels = np.concatenate([
        patch_np[:border, :, :3].reshape(-1, 3),
        patch_np[-border:, :, :3].reshape(-1, 3),
        patch_np[:, :border, :3].reshape(-1, 3),
        patch_np[:, -border:, :3].reshape(-1, 3)
    ])
    med = np.median(border_pixels, axis=0)
    dists = np.linalg.norm(border_pixels - med, axis=1)
    clean_pixels = border_pixels[dists < np.percentile(dists, 80)]
    bg_color = np.median(clean_pixels, axis=0)
    color_dist = np.linalg.norm(patch_np[:, :, :3].astype(np.float32) - bg_color, axis=2)
    threshold = np.percentile(color_dist, quantile)
    graffiti_mask = (color_dist > threshold).astype(np.uint8) * 255
    graffiti_mask = cv2.medianBlur(graffiti_mask, 5)
    graffiti_mask = cv2.threshold(graffiti_mask, 127, 255, cv2.THRESH_BINARY)[1]
    graffiti_mask = cv2.dilate(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)
    graffiti_mask = cv2.erode(graffiti_mask, np.ones((3, 3), np.uint8), iterations=1)
    return graffiti_mask

def extract_graffiti_rgba_from_patch(patch_np, graffiti_mask):
    alpha = graffiti_mask.astype(np.uint8)
    if patch_np.shape[2] == 3:
        rgba = np.dstack([patch_np, alpha])
    else:
        rgba = patch_np.copy()
        rgba[:, :, 3] = alpha
    rgba[alpha == 0] = [0, 0, 0, 0]
    return rgba

def match_color_statistics(src, target, blend_factor=0.5):
    """
    Matches the color statistics of the source (patch) to the target (ROI),
    with a blending factor to retain some of the original colors.
    
    Parameters:
        src: Source image (patch) as a NumPy array.
        target: Target image (ROI) as a NumPy array.
        blend_factor: Float between 0 and 1. Higher values retain more of the original colors.
    
    Returns:
        NumPy array with adjusted colors.
    """
    src_mean, src_std = src.mean(axis=(0, 1)), src.std(axis=(0, 1))
    tgt_mean, tgt_std = target.mean(axis=(0, 1)), target.std(axis=(0, 1))
    
    # Match color statistics
    matched = (src - src_mean) / (src_std + 1e-6) * (tgt_std + 1e-6) + tgt_mean
    
    # Blend with original colors
    blended = blend_factor * src + (1 - blend_factor) * matched
    return np.clip(blended, 0, 255)

def match_patch_color(patch_rgb, roi):
    return match_color_statistics(patch_rgb, roi)

def blur_to_match(patch_rgb, roi):
    roi_gray = cv2.cvtColor(roi.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    patch_gray = cv2.cvtColor(patch_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    roi_blur = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    patch_blur = cv2.Laplacian(patch_gray, cv2.CV_64F).var()
    if patch_blur > 0 and roi_blur > 0 and patch_blur > roi_blur:
        blur_ksize = int(np.clip((patch_blur / roi_blur) ** 0.5, 1, 5))
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        patch_rgb = cv2.GaussianBlur(patch_rgb, (blur_ksize, blur_ksize), 0)
    return patch_rgb

def add_realistic_noise(patch_rgb, std=2):
    noise = np.random.normal(0, std, patch_rgb.shape)
    return np.clip(patch_rgb + noise, 0, 255)

def seamless_blend(sign_image_np, patch_rgb, patch_alpha, patch_x_min, patch_y_min):
    mask = (patch_alpha * 255).astype(np.uint8)
    src_canvas = sign_image_np.copy()
    h, w = patch_rgb.shape[:2]
    for c in range(3):
        src_canvas[patch_y_min:patch_y_min + h, patch_x_min:patch_x_min + w, c] = (
            patch_alpha * patch_rgb[..., c] +
            (1 - patch_alpha) * src_canvas[patch_y_min:patch_y_min + h, patch_x_min:patch_x_min + w, c]
        )
    src_canvas = src_canvas.astype(np.uint8)
    mask_canvas = np.zeros(sign_image_np.shape[:2], dtype=np.uint8)
    mask_canvas[patch_y_min:patch_y_min + h, patch_x_min:patch_x_min + w] = 255
    center = (patch_x_min + w // 2, patch_y_min + h // 2)
    blended_image = cv2.seamlessClone(src_canvas, sign_image_np, mask_canvas, center, cv2.NORMAL_CLONE)
    return blended_image

def alpha_blend(roi, patch_rgb, patch_alpha, alpha):
    for c in range(3):
        roi[..., c] = (patch_alpha * patch_rgb[..., c] * alpha +
                       (1 - patch_alpha * alpha) * roi[..., c])
    return roi

def realistic_patch_applier(
    graffiti_rgba_patch_np,
    sign_image_np,
    target_bbox_on_sign,
    alpha=1.0,
    patch_scale=1.0,
    position_mode='center',
    x_offset=0.0,  # Used only in 'free' mode
    y_offset=0.0,  # Used only in 'free' mode
    use_seamless_clone=False,
    add_blur=True,
    add_noise=True
):
    if not (isinstance(target_bbox_on_sign, (list, tuple, np.ndarray)) and len(target_bbox_on_sign) == 4):
        raise ValueError(f"Invalid bbox: {type(target_bbox_on_sign)}, shape: {getattr(target_bbox_on_sign, 'shape', None)}, value: {target_bbox_on_sign}")
    sign_image_float = sign_image_np.astype(np.float32)
    patched_image_np = sign_image_float.copy()
    patch_x_min, patch_y_min, patch_width, patch_height = compute_patch_position(
        target_bbox_on_sign, patch_scale, position_mode,x_offset=x_offset, y_offset=y_offset
    )
    patch_x_max = patch_x_min + patch_width
    patch_y_max = patch_y_min + patch_height
    graffiti_resized = cv2.resize(graffiti_rgba_patch_np, (patch_width, patch_height), interpolation=cv2.INTER_AREA)
    patch_rgb = graffiti_resized[:, :, :3].astype(np.float32)
    patch_alpha = (graffiti_resized[:, :, 3].astype(np.float32) / 255.0)
    roi = patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3]
    patch_rgb = match_patch_color(patch_rgb, roi)
    if add_blur:
        patch_rgb = blur_to_match(patch_rgb, roi)
    if add_noise:
        patch_rgb = add_realistic_noise(patch_rgb)
    if use_seamless_clone:
        output = seamless_blend(sign_image_np, patch_rgb, patch_alpha, patch_x_min, patch_y_min)
        return output
    else:
        roi = alpha_blend(roi, patch_rgb, patch_alpha, alpha)
        patched_image_np[patch_y_min:patch_y_max, patch_x_min:patch_x_max, :3] = roi
        return patched_image_np.astype(np.uint8)