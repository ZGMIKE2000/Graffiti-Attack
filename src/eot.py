import kornia.augmentation as K
import torch
import numpy as np

# Define the augmentation pipeline
augmentation_pipeline = torch.nn.Sequential(
    # Geometric transformations
    K.RandomRotation(degrees=30.0),  # Random rotation up to ±30 degrees
    K.RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Affine transformations
    K.RandomPerspective(distortion_scale=0.5, p=0.5),  # Perspective distortion

    # Photometric transformations
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
    K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),  # Add Gaussian noise
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),  # Add blur

    # Environmental effects (optional)
    K.RandomSolarize(threshold=0.5, p=0.3),  # Simulate strong lighting
    K.RandomMotionBlur(kernel_size=5, angle=10.0, direction=0.5, p=0.3)  # Simulate motion blur
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
    augmented_patches = []
    for _ in range(num_samples):
        augmented_patch = augmentation_pipeline(patch.unsqueeze(0))  # Apply augmentations
        augmented_patches.append(augmented_patch)
    return torch.cat(augmented_patches, dim=0)  # Return as a batch

# # Example usage
# patch_np = np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8)  # Example patch
# patch_tensor = torch.tensor(patch_np / 255.0, dtype=torch.float32)  # Normalize to [0, 1]

# augmented_patches = apply_eot(patch_tensor, num_samples=10)
# print(f"Generated {augmented_patches.shape[0]} augmented patches.")