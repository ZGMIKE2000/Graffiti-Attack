import kornia.augmentation as K
import torch

# Define the augmentation pipeline
augmentation_pipeline = torch.nn.Sequential(
    # Geometric transformations
    K.RandomRotation(degrees=30.0),  # Simulate rotation
    K.RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(1.0, 1.0)),  # Affine transformations
    K.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective distortion

    # Photometric transformations
    K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Lighting variations
    K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),  # Add noise
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),  # Add blur

    # Environmental effects
    K.RandomMotionBlur(kernel_size=5, angle=10.0, direction=0.5, p=0.3),  # Simulate motion blur
)

def apply_eot(patch, num_samples=10):
    """
    Apply EOT augmentations to the adversarial patch.
    
    Args:
        patch (torch.Tensor): The adversarial patch (C, H, W).
        num_samples (int): Number of augmented samples to generate.
    
    Returns:
        torch.Tensor: A batch of augmented patches.
    """

    rgb_patch = patch[:, :3, :, :]  # Extract RGB channels
    alpha_channel = patch[:, 3:, :, :]  # Extract alpha channel
    augmented_patches = []
    for _ in range(num_samples):
        augmented_rgb_patch = augmentation_pipeline(rgb_patch)
        augmented_rgba_patch = torch.cat([augmented_rgb_patch, alpha_channel], dim=1)
        augmented_patches.append(augmented_rgba_patch)
    return torch.cat(augmented_patches, dim=0)  # Return as a batch
