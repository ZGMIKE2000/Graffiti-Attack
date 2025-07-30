import os
import cv2
import torch
import logging
import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.decomposition import PCA
from optuna import optimize_patch_placement

from patch import (
    extract_graffiti_mask_from_patch,
    extract_graffiti_rgba_from_patch,
    realistic_patch_applier,
)
from eot import apply_eot  # Import the EOT function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def evaluate_patch_effectiveness_single(
    yolov8_model, patched_image_np: np.ndarray, target_class_id: int
) -> float:
    """
    Evaluate the effectiveness of a patch for a single target class.

    Args:
        yolov8_model: YOLOv8 model for evaluation.
        patched_image_np (np.ndarray): Patched image.
        target_class_id (int): Target class ID.

    Returns:
        float: Highest confidence for the target class (lower is better).
    """
    results = yolov8_model(patched_image_np, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 0.0  # No detections = perfect fooling

    res = results[0].boxes
    return max(
        (float(res.conf[i]) for i, _ in enumerate(res.xyxy) if int(res.cls[i]) == target_class_id),
        default=0.0,
    )

def evaluate_patch_effectiveness_multi(
    yolov8_model, patched_image_np: np.ndarray, target_class_ids: List[int]
) -> float:
    """
    Evaluate the effectiveness of a patch for multiple target classes.

    Args:
        yolov8_model: YOLOv8 model for evaluation.
        patched_image_np (np.ndarray): Patched image.
        target_class_ids (List[int]): List of target class IDs.

    Returns:
        float: Average confidence for all target classes (lower is better).
    """
    results = yolov8_model(patched_image_np, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 0.0  # No detections = perfect fooling

    res = results[0].boxes
    confidences = [
        max(
            (float(res.conf[i]) for i, _ in enumerate(res.xyxy) if int(res.cls[i]) == target_class_id),
            default=0.0,
        )
        for target_class_id in target_class_ids
    ]
    return np.mean(confidences) if confidences else 0.0

def generate_patch(generator_model, latent_np: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a patch image and mask from a latent vector.

    Args:
        generator_model: StyleGAN3 generator model.
        latent_np (np.ndarray): Latent vector.
        device (torch.device): Device to run the generator on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated patch image and mask.
    """
    latent = torch.tensor(latent_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        gen_img_tensor = generator_model(latent, None, truncation_psi=1, noise_mode='random')
    gen_img_np = (gen_img_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    gen_mask_np = extract_graffiti_mask_from_patch(gen_img_np)
    gen_mask_np = extract_graffiti_rgba_from_patch(gen_img_np, gen_mask_np)
    return gen_img_np, gen_mask_np

def apply_patch_to_dataset(
    gen_mask_np: np.ndarray,
    dataset: List[Tuple[np.ndarray, List]],
    yolov8_model,
    baseline_confidences: List[float],
    target_class_ids: List[int],
    num_eot_samples: int = 10
) -> List[float]:
    """
    Apply a patch to all images in a dataset and compute losses.

    Args:
        gen_mask_np (np.ndarray): Generated patch mask.
        dataset (List[Tuple[np.ndarray, List]]): Dataset of images and bounding boxes.
        yolov8_model: YOLOv8 model for evaluation.
        baseline_confidences (List[float]): Baseline confidences for the dataset.
        target_class_ids (List[int]): List of target class IDs.
        num_eot_samples (int): Number of EOT augmentations to apply.

    Returns:
        List[float]: Losses for each image in the dataset.
    """
    losses = []
    gen_mask_tensor = torch.tensor(gen_mask_np / 255.0, dtype=torch.float32)  # Normalize to [0, 1]
    augmented_patches = apply_eot(gen_mask_tensor, num_samples=num_eot_samples)

    for idx, (img_np, bboxes, *_) in enumerate(dataset):
        patched_image_np = img_np.copy()
        image_losses = []

        for augmented_patch in augmented_patches:
            augmented_patch_np = (augmented_patch.squeeze(0).numpy() * 255).astype(np.uint8)
            patched_image = patched_image_np.copy()

            for bbox in bboxes:
                patched_image = realistic_patch_applier(
                    augmented_patch_np,
                    patched_image,
                    bbox,
                    alpha=0.8,
                    patch_scale=0.5,
                    position_mode='center',
                    use_seamless_clone=False,
                    add_blur=True,
                    add_noise=True
                )
            
            patched_conf = evaluate_patch_effectiveness_single(yolov8_model, patched_image, target_class_ids)
            baseline_conf = baseline_confidences[idx]
            loss = patched_conf - baseline_conf
            image_losses.append(loss)

        losses.append(np.mean(image_losses))  # Aggregate losses (e.g., mean or max)

    return losses

def apply_patch_to_dataset_optuna(
    gen_mask_np: np.ndarray,
    dataset: List[Tuple[np.ndarray, List]],
    yolov8_model,
    baseline_confidences: List[float],
    target_class_ids: List[int],
    num_eot_samples: int = 10
) -> List[float]:
    optimized_params = optimize_patch_placement(
        dataset=dataset,
        graffiti_rgba_patch_np=gen_mask_np,
        yolov8_model=yolov8_model,
        target_class_id=target_class_ids[0]  # Assuming single target class for simplicity
    )

    losses = []
    gen_mask_tensor = torch.tensor(gen_mask_np / 255.0, dtype=torch.float32)  # Normalize to [0, 1]
    augmented_patches = apply_eot(gen_mask_tensor, num_samples=num_eot_samples)

    for idx, (img_np, bboxes, *_) in enumerate(dataset):
        patched_image_np = img_np.copy()
        image_losses = []

        for bbox_index, bbox in enumerate(bboxes):
            bbox_key = (idx, bbox_index)  # Unique key for each bounding box
            best_params = optimized_params[bbox_key]["best_params"]
            alpha = best_params["alpha"]
            patch_scale = best_params["patch_scale"]
            x_offset = best_params["x_offset"]
            y_offset = best_params["y_offset"]

            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
            bbox_width = bbox_x_max - bbox_x_min
            bbox_height = bbox_y_max - bbox_y_min
            patch_x_min = int(bbox_x_min + x_offset * bbox_width)
            patch_y_min = int(bbox_y_min + y_offset * bbox_height)

            for augmented_patch in augmented_patches:
                augmented_patch_np = (augmented_patch.squeeze(0).numpy() * 255).astype(np.uint8)

                patched_image = realistic_patch_applier(
                    augmented_patch_np,
                    patched_image_np.copy(),  # Use a copy to avoid overwriting
                    bbox,
                    alpha=alpha,
                    patch_scale=patch_scale,
                    position_mode="custom",
                    custom_position=(patch_x_min, patch_y_min),
                    use_seamless_clone=False,
                    add_blur=True,
                    add_noise=True
                )

                patched_conf = evaluate_patch_effectiveness_single(yolov8_model, patched_image, target_class_ids[0])
                baseline_conf = baseline_confidences[idx][bbox_index]  # Ensure correct indexing
                loss = patched_conf - baseline_conf
                image_losses.append(loss)

        losses.append(np.mean(image_losses))  # Aggregate losses for the image

    return losses


def save_checkpoint(
    gen_img_np: np.ndarray,
    latent_np: np.ndarray,
    output_dir: str,
    call_count: int,
    max_loss: float,
    best_loss: float,
    is_best: bool = False
) -> None:
    """
    Save the current patch and latent vector as a checkpoint.

    Args:
        gen_img_np (np.ndarray): Generated patch image.
        latent_np (np.ndarray): Latent vector.
        output_dir (str): Directory to save checkpoints.
        call_count (int): Current evaluation count.
        max_loss (float): Current loss.
        best_loss (float): Best loss so far.
        is_best (bool): Whether this is the best checkpoint.
    """
    patch_path = os.path.join(output_dir, f"graffiti_patch_eval_{call_count}.png")
    cv2.imwrite(patch_path, gen_img_np)
    latent_path = os.path.join(output_dir, f"latent_eval_{call_count}.npy")
    np.save(latent_path, latent_np)
    if is_best:
        logging.info(f"New best loss: {best_loss:.4f} at eval {call_count}")
    else:
        logging.info(f"Eval {call_count}: Current loss {max_loss:.4f}, Best loss {best_loss:.4f}")
        logging.info(f"Saved graffiti patch at eval {call_count}")