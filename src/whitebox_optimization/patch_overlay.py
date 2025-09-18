import torch
import torch.nn.functional as F

def overlay_patch_rgba(sign_image, patch_rgba, bbox, alpha=1.0):
    """
    Overlays an RGBA patch onto a batch of images at specified bounding boxes.

    Args:
        sign_image (Tensor): (B, 3, H, W) input images.
        patch_rgba (Tensor): (B, 4, h, w) RGBA patches (A in [0,1]).
        bbox (Tensor): (B, 4) bounding boxes [x_min, y_min, x_max, y_max] (pixel coords).
        alpha (float): Global blending factor (default 1.0).

    Returns:
        Tensor: (B, 3, H, W) images with patch overlaid.
    """
    B, _, H, W = sign_image.shape
    out_imgs = []
    for i in range(B):
        x_min, y_min, x_max, y_max = bbox[i].long()
        pw, ph = x_max - x_min, y_max - y_min

        # Split patch into RGB and alpha
        patch_rgb = patch_rgba[i, :3].unsqueeze(0)
        patch_a = patch_rgba[i, 3:].unsqueeze(0)

        # Resize patch and alpha to bbox size
        patch_resized = F.interpolate(patch_rgb, size=(ph, pw), mode="bicubic", align_corners=False).squeeze(0)
        alpha_resized = F.interpolate(patch_a, size=(ph, pw), mode="bicubic", align_corners=False).squeeze(0)
        patch_alpha = alpha_resized * alpha

        img = sign_image[i].clone()
        roi = img[:, y_min:y_max, x_min:x_max]
        blended = patch_alpha * patch_resized + (1 - patch_alpha) * roi

        # Place blended region back into image
        new_img = img.clone()
        new_img[:, y_min:y_max, x_min:x_max] = blended
        out_imgs.append(new_img)
    return torch.stack(out_imgs, dim=0)