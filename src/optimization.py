import os
import cv2
import torch
import logging
import nevergrad as ng
import numpy as np
from typing import List, Tuple

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

def evaluate_patch_effectiveness_batch(
    yolov8_model, patched_images: list, target_class_ids: list
) -> list:
    """
    Evaluate a batch of patched images for the highest confidence of any target class.
    Returns a list of confidences (one per image).
    """
    # Ensure target_class_ids is always a list
    if isinstance(target_class_ids, int):
        target_class_ids = [target_class_ids]
        
    results = yolov8_model(patched_images, verbose=False)
    confidences = []
    for res in results:
        if not hasattr(res, "boxes") or res.boxes is None:
            confidences.append(0.0)
        else:
            boxes = res.boxes
            conf = max(
                (float(boxes.conf[i]) for i, _ in enumerate(boxes.xyxy)
                 if int(boxes.cls[i]) in target_class_ids),
                default=0.0,
            )
            confidences.append(conf)
    return confidences

def compute_baseline_confidences(
    yolov8_model, 
    dataset: List[Tuple[np.ndarray, List]], 
    target_class_id: int
) -> List[float]:
    """
    Compute baseline confidences for the target class(es) in the unpatched images.

    Args:
        yolov8_model: YOLOv8 model for evaluation.
        dataset (List[Tuple[np.ndarray, List]]): Dataset of images and bounding boxes.
        target_class_ids (List[int]): List of target class IDs.

    Returns:
        List[float]: Baseline confidences for each image in the dataset.
    """
    baseline_confidences = []
    for idx, (img_np, _, *_) in enumerate(dataset):
        # Run YOLO on the unpatched image
        confidence = evaluate_patch_effectiveness_single(
            yolov8_model=yolov8_model,
            patched_image_np=img_np,
            target_class_id=target_class_id
        )
        baseline_confidences.append(confidence)
        logging.info(f"Image {idx}: Baseline confidence = {confidence:.4f}")
    return baseline_confidences

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

def evaluate_patch_eot(
    gen_mask_np: np.ndarray,
    dataset: List[Tuple[np.ndarray, List]],
    yolov8_model,
    baseline_confidences: List[float],
    target_class_ids: List[int],
    alpha: float = 1.0,
    patch_scale: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    num_eot_samples: int = 5
) -> List[float]:
    """
    Apply a patch to all images in a dataset and compute losses using optimized positioning parameters.
    Uses batch YOLOv8 inference for speed.
    """
    losses = []
    gen_mask_tensor = torch.tensor(gen_mask_np / 255.0, dtype=torch.float32)
    gen_mask_tensor = gen_mask_tensor.unsqueeze(0)
    augmented_patches = apply_eot(gen_mask_tensor, num_samples=num_eot_samples)

    for idx, (img_np, bboxes, *_) in enumerate(dataset):
        patched_images = []
        for bbox in bboxes:
            for augmented_patch in augmented_patches:
                augmented_patch_np = (augmented_patch.squeeze(0).numpy() * 255).astype(np.uint8)
                patched_image = realistic_patch_applier(
                    graffiti_rgba_patch_np=augmented_patch_np,
                    sign_image_np=img_np.copy(),
                    target_bbox_on_sign=bbox,
                    alpha=alpha,
                    patch_scale=patch_scale,
                    position_mode="free",
                    x_offset=x_offset,
                    y_offset=y_offset,
                    use_seamless_clone=False,
                    add_blur=True,
                    add_noise=True
                )
                patched_images.append(patched_image)

        # Batch inference
        if patched_images:
            batch_confidences = evaluate_patch_effectiveness_batch(yolov8_model, patched_images, target_class_ids)
            image_losses = []
            for patched_conf in batch_confidences:
                baseline_conf = baseline_confidences[idx]
                loss = patched_conf - baseline_conf
                image_losses.append(loss)
            losses.append(np.mean(image_losses))
        else:
            losses.append(0.0)

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

def is_significantly_different(latent1, latent2, scale1, scale2, pos1, pos2, latent_thresh=15.0, scale_thresh=0.05, pos_thresh=5):
    # Euclidean distance for latent code
    latent_diff = np.linalg.norm(np.array(latent1) - np.array(latent2))
    print("Latent code difference:", latent_diff)
    scale_diff = abs(scale1 - scale2)
    pos_diff = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return (latent_diff > latent_thresh) or (scale_diff > scale_thresh) or (pos_diff > pos_thresh)

def run_es_optimization(
    generator_model, yolov8_model, real_sign_images_dataset,
    target_class_ids, num_generations, population_size, checkpoint_dir
):
    """
    Runs evolutionary strategy (Nevergrad) to find a patch that fools YOLOv8.
    Saves checkpoints every 100 evaluations and whenever a new best loss is found.
    Returns the best latent code and its loss.

    Args:
        generator_model: StyleGAN3 generator model.
        yolov8_model: YOLOv8 model for evaluation.
        real_sign_images_dataset: Dataset of images and bounding boxes.
        target_class_ids: List of target class IDs for the attack.
        num_generations: Number of generations for the ES optimization.
        population_size: Number of individuals in the population.
        output_dir: Directory to save checkpoints.

    Returns:
        Tuple: Best latent code and its corresponding loss.
    """
    device = next(generator_model.parameters()).device
    latent_dim = getattr(generator_model, 'z_dim', 
                 getattr(generator_model.mapping, 'z_dim', 
                 getattr(generator_model.mapping, 'w_dim', 512)))

    # Ensure the output directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)


    # Compute baseline confidences
    logging.info("Computing baseline confidences...")
    baseline_confidences = compute_baseline_confidences(
        yolov8_model=yolov8_model,
        dataset=real_sign_images_dataset,
        target_class_id=target_class_ids
    )
    logging.info(f"Baseline confidences: {baseline_confidences}")

    # search_space = ng.p.Array(shape=(latent_dim,)).set_bounds(-3, 3)  # Latent code bounds
    # Define the search space (latent code + positioning parameters)
    latent_bounds = (-3, 3)  # Bounds for latent code
    alpha_bounds = (0.5, 1.0)    # Bounds for alpha (transparency)
    offset_bounds = (-50, 50)  # Example bounds for x_offset and y_offset (adjust based on image dimensions)
    scale_bounds = (0.4,0.6)  # Bounds for patch_scale (adjust based on your requirements)

    # Create a tuple for the search space
    instrumentation = ng.p.Tuple(
        ng.p.Array(shape=(latent_dim,)).set_bounds(*latent_bounds).set_mutation(sigma=1.0),  # Latent code
        ng.p.Scalar(init=0.75).set_bounds(*alpha_bounds).set_mutation(sigma=0.1),                                   # Alpha (init=0.75 within 0.5-1.0)
        ng.p.Scalar(init=0.0).set_bounds(*offset_bounds).set_mutation(sigma=2),                                  # x_offset (init=0.0 within -50 to 50)
        ng.p.Scalar(init=0.0).set_bounds(*offset_bounds).set_mutation(sigma=2),                                  # y_offset (init=0.0 within -50 to 50)
        ng.p.Scalar(init=0.5).set_bounds(*scale_bounds).set_mutation(sigma=0.05)                                  # patch_scale (init=0.5 within 0.2-1.0)
    )

    def objective(params):
        logging.info(f"Starting evaluation {objective.call_count + 1}...")
        latent_np, alpha, x_offset, y_offset, patch_scale = params
        _, gen_mask_np = generate_patch(generator_model, latent_np, device)
        losses = evaluate_patch_eot(
            gen_mask_np=gen_mask_np,
            dataset=real_sign_images_dataset,
            yolov8_model=yolov8_model,
            baseline_confidences=baseline_confidences,
            target_class_ids=target_class_ids,
            alpha=alpha,
            patch_scale=patch_scale,
            x_offset=x_offset,
            y_offset=y_offset,
            num_eot_samples=5
        )
        current_loss = np.mean(losses)


        # save if the best loss patch's latent code is different from the previously saved one or if its not the best loss path, save only if the loss improved
        if np.allclose(losses, -np.array(baseline_confidences), atol=1e-4):
            should_save = (objective.last_saved_latent is None or 
            is_significantly_different(
                latent_np, objective.last_saved_latent,
                patch_scale, objective.last_saved_scale,
                (x_offset, y_offset), objective.last_saved_pos
                )    
            )
        else:
            logging.info(f"current_loss: {current_loss}, objective best loss:{objective.best_loss}")
            should_save = (abs(current_loss)-abs(objective.best_loss) >= 0.02 )

        if should_save:
            logging.info(f"wtf")
            gen_img_np, gen_mask_np = generate_patch(generator_model, latent_np, device)
            save_checkpoint(
                gen_img_np=gen_img_np,
                latent_np=latent_np,
                output_dir=checkpoint_dir,
                call_count=objective.call_count,
                max_loss=current_loss,
                best_loss=current_loss,
                is_best=True
            )
            # Update last saved
            objective.last_saved_latent = np.copy(latent_np)
            objective.last_saved_scale = patch_scale
            objective.last_saved_pos = (x_offset, y_offset)

        # Always update best_loss if a better loss is found
        if current_loss < objective.best_loss:
            objective.best_loss = current_loss

        # Log the parameters and loss to a CSV file
        log_file_path = os.path.join(checkpoint_dir, "optimization_log.csv")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{objective.call_count + 1},alpha:{alpha},x:{x_offset},y:{y_offset},scale:{patch_scale},loss:{current_loss}\n")

        objective.call_count += 1
        return current_loss

    # Initialize counters for saving checkpoints
    objective.call_count = 0
    objective.best_loss = float(0)

    if not hasattr(objective, "last_saved_latent"):
        objective.last_saved_latent = None
        objective.last_saved_scale = None
        objective.last_saved_pos = None

    # Define the instrumentation for Nevergrad
    optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=num_generations * population_size, num_workers=1)
    recommendation = optimizer.minimize(objective)
    
    best_params = recommendation.value

    # Generate and save the patch for these params
    gen_img_np, _ = generate_patch(generator_model, best_params[0], device)
    save_checkpoint(
        gen_img_np=gen_img_np,
        latent_np=best_params[0],
        output_dir=checkpoint_dir,
        call_count="final_recommendation",
        max_loss=0.0,
        best_loss=objective.best_loss,
        is_best=True
    )