import nevergrad as ng
import numpy as np
import torch

from patch import *


def evaluate_patch_effectiveness_multi(yolov8_model, patched_image_np, target_class_ids):
    """
    Returns average confidence for all target classes present in the detector output.
    Lower is better (0 = none detected).
    """
    results = yolov8_model(patched_image_np, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 0.0  # No detections = perfect fooling

    res = results[0].boxes
    total_conf = 0.0
    count = 0
    for target_class_id in target_class_ids:
        found = False
        best_conf = 0.0
        for i, box in enumerate(res.xyxy):
            class_id = int(res.cls[i])
            confidence = float(res.conf[i])
            if class_id == target_class_id:
                found = True
                best_conf = max(best_conf, confidence)
        if found:
            total_conf += best_conf
            count += 1
    # If none of the target classes are detected, return 0.0
    return total_conf / count if count > 0 else 0.0

def evaluate_patch_misclassification(yolov8_model, patched_image_np, original_class_id, target_class_id):
    """
    Evaluates if the patch causes YOLOv8 to misclassify the original class as the target class.
    Returns a loss: lower is better (0 = original class not detected, target class detected).
    """
    results = yolov8_model(patched_image_np, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 1.0  # No detections at all: not a successful misclassification

    res = results[0].boxes
    found_original = False
    found_target = False
    target_confidence = 0.0

    for i, box in enumerate(res.xyxy):
        class_id = int(res.cls[i])
        confidence = float(res.conf[i])
        if class_id == original_class_id:
            found_original = True
        if class_id == target_class_id:
            found_target = True
            target_confidence = max(target_confidence, confidence)

    # Loss logic:
    # - If original class is detected: high loss
    # - If target class is detected (and original is not): low loss (reward)
    if found_original:
        return 1.0  # Penalize if original class is still detected
    elif found_target:
        return 1.0 - target_confidence  # Lower loss for higher confidence in target class
    else:
        return 1.0  # Neither detected: not a successful targeted attack

def run_es_optimization(generator_model, yolov8_model, real_sign_images_dataset,
                       target_class_ids, num_generations=100, population_size=10):
    """
    Runs evolutionary strategy (Nevergrad) to find a patch that fools YOLOv8.
    Returns the best latent code and its loss.
    """
    device = next(generator_model.parameters()).device

    # Determine latent dimension
    latent_dim = getattr(generator_model, 'z_dim', 
                 getattr(generator_model.mapping, 'z_dim', 
                 getattr(generator_model.mapping, 'w_dim', 512)))
    
    def objective(latent_np):
        latent = torch.tensor(latent_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            gen_img_tensor = generator_model(latent, None, truncation_psi=1, noise_mode='const')
        gen_img_np = (gen_img_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0,255).astype(np.uint8)
        gen_mask_np = extract_graffiti_mask_from_patch(gen_img_np)
        gen_mask_np = extract_graffiti_rgba_from_patch(gen_img_np, gen_mask_np)
        total_loss = 0.0
        for img_np, bbox, *_ in real_sign_images_dataset:
            patched_image_np = realistic_patch_applier(gen_mask_np, img_np, bbox)
            loss = evaluate_patch_effectiveness_multi(yolov8_model, patched_image_np, target_class_ids)
            # print("DEBUG: Patch loss (mean confidence for target classes):", loss)
            total_loss += loss
        return total_loss / max(1, len(real_sign_images_dataset))

    instrumentation = ng.p.Array(shape=(latent_dim,)).set_bounds(-3, 3)
    optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=num_generations * population_size, num_workers=1)
    recommendation = optimizer.minimize(objective)
    best_latent_code = torch.tensor(recommendation.value, dtype=torch.float32, device=device).unsqueeze(0)
    best_loss = objective(recommendation.value)
    return best_latent_code, best_loss




# def run_es_optimization(generator_model, yolov8_model, real_sign_images_dataset,
#                        target_class_ids, num_generations=100, population_size=10):
#     """
#     Runs evolutionary strategy (Nevergrad) to find a patch that fools YOLOv8.
#     Returns the best latent code and its loss.
#     """
#     device = next(generator_model.parameters()).device

#     # Determine latent dimension
#     latent_dim = getattr(generator_model, 'z_dim', 
#                  getattr(generator_model.mapping, 'z_dim', 
#                  getattr(generator_model.mapping, 'w_dim', 512)))
    
#     def objective(latent_np):
#         latent = torch.tensor(latent_np, dtype=torch.float32, device=device).unsqueeze(0)
#         with torch.no_grad():
#             gen_img_tensor = generator_model(latent, None, truncation_psi=1, noise_mode='const')
#         gen_img_np = (gen_img_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0,255).astype(np.uint8)
#         gen_mask_np = extract_graffiti_mask_from_patch(gen_img_np)
#         gen_mask_np = extract_graffiti_rgba_from_patch(gen_img_np, gen_mask_np)
#         total_loss = 0.0
#         for img_np, bbox, *_ in real_sign_images_dataset:
#             patched_image_np = realistic_patch_applier(gen_mask_np, img_np, bbox)
#             loss = evaluate_patch_effectiveness_multi(yolov8_model, patched_image_np, target_class_ids)
#             # print("DEBUG: Patch loss (mean confidence for target classes):", loss)
#             total_loss += loss
#         return total_loss / max(1, len(real_sign_images_dataset))

#     instrumentation = ng.p.Array(shape=(latent_dim,)).set_bounds(-3, 3)
#     optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=num_generations * population_size, num_workers=1)
#     recommendation = optimizer.minimize(objective)
#     best_latent_code = torch.tensor(recommendation.value, dtype=torch.float32, device=device).unsqueeze(0)
#     best_loss = objective(recommendation.value)
#     return best_latent_code, best_loss