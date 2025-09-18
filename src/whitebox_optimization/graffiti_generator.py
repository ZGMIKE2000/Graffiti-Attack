import torch
import numpy as np
import torchvision.utils as vutils

class GraffitiPatchGenerator:
    """
    Wrapper for a StyleGAN3 generator to produce adversarial patches and masks.
    Provides utilities for mask extraction, RGBA conversion, and saving.
    """
    def __init__(self, G, device="cuda"):
        """
        Args:
            G: StyleGAN3 generator (PyTorch module)
            device: Device string, e.g. "cuda" or "cpu"
        """
        self.G = G
        self.device = device

    def generate(self, z):
        """
        Generate a patch from latent vector z.
        If output is in [-1, 1], shift to [0, 1].
        """
        patch = self.G(z, None)
        if patch.min() < 0:
            patch = patch * 0.5 + 0.5
        return patch

    def _extract_mask(self, patch, border=10, quantile=0.85, sharpness=20.0):
        """
        Extract a mask from the patch using color distance from border pixels.
        Args:
            patch: (B, 3, H, W) tensor
            border: Border width in pixels
            quantile: Quantile for thresholding
            sharpness: Controls mask sharpness
        Returns:
            mask: (B, 1, H, W) tensor
        """
        B, C, H, W = patch.shape
        # Collect border pixels
        top    = patch[:, :, :border, :].reshape(B, C, -1)
        bottom = patch[:, :, -border:, :].reshape(B, C, -1)
        left   = patch[:, :, :, :border].reshape(B, C, -1)
        right  = patch[:, :, :, -border:].reshape(B, C, -1)
        border_pixels = torch.cat([top, bottom, left, right], dim=2)
        bg_color = border_pixels.median(dim=2).values
        # Compute distance from background color
        patch_flat = patch.permute(0,2,3,1).reshape(-1,3)
        bg_expand = bg_color.repeat_interleave(H*W, dim=0)
        dists = torch.norm(patch_flat - bg_expand, dim=1)
        k = int(dists.numel() * quantile)
        thresh = torch.topk(dists, k).values.min()
        mask = torch.sigmoid((dists - thresh) * sharpness)
        mask = mask.view(B, H, W).unsqueeze(1)
        return mask

    def auto_quantile_mask(self, patch, border=10, sharpness=20.0, quantile_range=(0.05, 0.95), steps=20):
        """
        Automatically select a quantile for mask extraction based on border/center separation.
        Returns:
            best_mask: (B, 1, H, W) tensor
            best_quantile: float
        """
        best_quantile = None
        best_score = -float('inf')
        best_mask = None
        B, C, H, W = patch.shape
        for q in np.linspace(*quantile_range, steps):
            mask = self._extract_mask(patch, border=border, quantile=q, sharpness=sharpness)
            border_mask = torch.cat([
                mask[:, :, :border, :].reshape(B, -1),
                mask[:, :, -border:, :].reshape(B, -1),
                mask[:, :, :, :border].reshape(B, -1),
                mask[:, :, :, -border:].reshape(B, -1)
            ], dim=1)
            center_mask = mask[:, :, border:-border, border:-border].reshape(B, -1)
            border_mean = border_mask.mean().item()
            center_mean = center_mask.mean().item()
            score = center_mean - border_mean * 2
            if border_mean < 0.1 and score > best_score:
                best_score = score
                best_quantile = q
                best_mask = mask
        return best_mask, best_quantile

    def _to_rgba(self, patch, mask):
        """
        Combine patch and mask into an RGBA tensor.
        Args:
            patch: (B, 3, H, W)
            mask: (B, 1, H, W)
        Returns:
            rgba: (B, 4, H, W)
        """
        patch_fg = patch * mask
        rgba = torch.cat([patch_fg, mask], dim=1)
        return rgba

    def generate_rgba(self, z, border=10, sharpness=20.0, quantile_range=(0.05, 0.95), steps=20):
        """
        Generate an RGBA patch from latent z.
        Returns:
            rgba: (B, 4, H, W)
            patch: (B, 3, H, W)
            mask: (B, 1, H, W)
            best_q: float
        """
        patch = self.generate(z)
        mask, best_q = self.auto_quantile_mask(patch, border=border, sharpness=sharpness, quantile_range=quantile_range, steps=steps)
        rgba = self._to_rgba(patch, mask)
        return rgba, patch, mask, best_q
    
    @staticmethod
    def save_patch(patch_tensor, step):
        """
        Save a patch tensor as an image for inspection.
        Args:
            patch_tensor: (B, 3, H, W) or (3, H, W)
            step: int, step number for filename
        """
        patch = patch_tensor.detach().cpu()
        if patch.dim() == 4:
            patch = patch[0]
        vutils.save_image(patch, f"patch_step_{step}.png", normalize=True)