import os
import logging
import cv2
from PIL import Image
import torch
from patch import extract_graffiti_mask_from_patch, extract_graffiti_rgba_from_patch, realistic_patch_applier
from utils import get_classes_for_image, yolo_bbox_to_pixel, get_bbox_for_image
from optimization import run_es_optimization, evaluate_patch_effectiveness_multi, evaluate_patch_misclassification
import numpy as np

def process_image(
    G_frozen, yolov8_model, image_path, bbox_yolo , bbox_folder, target_class_id, target_misclassify_id,
    num_generations, population_size, output_dir
):
    """
    Run the adversarial patch optimization and evaluation for a single image.

    Args:
        G_frozen: The loaded StyleGAN3 generator.
        yolov8_model: The loaded YOLOv8 model.
        image_path (str): Path to the input image.
        bbox_yolo (list): YOLO-format bounding box [x_center, y_center, w, h].
        target_class_id (int): The class ID to attack.
        target_misclassify_id (int or None): The class ID to induce misclassification to.
        num_generations (int): Number of generations for optimization.
        population_size (int): Population size for optimization.
        output_dir (str): Directory to save results.
    """
    if not os.path.exists(image_path):
        logging.error(f"Sign image not found at {image_path}")
        return
    
    
    present_class_ids = get_classes_for_image(image_path, bbox_folder=bbox_folder)

    sign_image_np = cv2.imread(image_path)
    img_h, img_w = sign_image_np.shape[:2]
    target_bbox_on_sign = yolo_bbox_to_pixel(bbox_yolo, img_w, img_h)
    real_sign_images_dataset = [(sign_image_np, target_bbox_on_sign)]

    logging.info(f"Processing image: {image_path}")
    results = yolov8_model(sign_image_np, verbose=True)
    if hasattr(results[0], "boxes"):
        for i, box in enumerate(results[0].boxes.xyxy):
            class_id = int(results[0].boxes.cls[i])
            confidence = float(results[0].boxes.conf[i])
            logging.info(f"Detection: class_id={class_id}, confidence={confidence}, box={box.cpu().numpy()}")

    # --- Run Optimization ---
    logging.info("Starting Black-Box Patch Optimization...")
    best_latent_code, final_best_loss = run_es_optimization(
        G_frozen, yolov8_model, real_sign_images_dataset,
        present_class_ids,  # Pass the list of present classes
        num_generations=num_generations, population_size=population_size
    )

    if best_latent_code is not None:
        logging.info(f"Optimization finished. Best Loss: {final_best_loss:.4f}")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

    with torch.no_grad():
        best_patch_tensor = G_frozen(best_latent_code, c=None, truncation_psi=0.7)
    best_patch_np = (best_patch_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    best_mask_np = extract_graffiti_mask_from_patch(best_patch_np)
    best_rgba_patch_np = extract_graffiti_rgba_from_patch(best_patch_np, best_mask_np)

    # REMOVE this line (no saving in src folder)
    # cv2.imwrite(f"{os.path.splitext(os.path.basename(image_path))[0]}_before.png", sign_image_np)

    final_patched_image_np = realistic_patch_applier(
        best_rgba_patch_np, sign_image_np, target_bbox_on_sign
    )

    # Save everything in the image's output subfolder
    Image.fromarray(best_rgba_patch_np).save(
        os.path.join(image_output_dir, "best_graffiti_patch_rgba.png")
    )
    Image.fromarray(best_mask_np).save(
        os.path.join(image_output_dir, "best_graffiti_mask.png")
    )
    cv2.imwrite(os.path.join(image_output_dir, "before.png"), sign_image_np)
    Image.fromarray(cv2.cvtColor(final_patched_image_np, cv2.COLOR_BGR2RGB)).save(
        os.path.join(image_output_dir, "final_adversarial_patched_sign.png")
    )
    Image.fromarray(cv2.cvtColor(best_patch_np, cv2.COLOR_BGR2RGB)).save(
        os.path.join(image_output_dir, "best_graffiti_patch.png")
    )

    latent_code_path = os.path.join(image_output_dir, "best_latent_code.pt")
    torch.save(best_latent_code, latent_code_path)
    logging.info(f"Best latent code saved to {latent_code_path}")

    logging.info("Best patch and final patched sign saved.")

    # Re-evaluate YOLOv8 on the final image
    logging.info("Re-evaluating final patched sign with YOLOv8:")
    final_results = yolov8_model(final_patched_image_np, verbose=True)
    if hasattr(final_results[0], "boxes"):
        for i, box in enumerate(final_results[0].boxes.xyxy):
            class_id = int(final_results[0].boxes.cls[i])
            confidence = float(final_results[0].boxes.conf[i])
            logging.info(f"AFTER PATCH: class_id={class_id}, confidence={confidence}, box={box.cpu().numpy()}")
    logging.info(final_results)
    final_loss = evaluate_patch_effectiveness_multi(yolov8_model, final_patched_image_np, present_class_ids)
    logging.info(f"Final YOLOv8 loss (confidence for target class) on patched image: {final_loss:.4f}")

    if target_misclassify_id is not None:
        misclass_loss = evaluate_patch_misclassification(
            yolov8_model, final_patched_image_np, target_class_id, target_misclassify_id
        )
        logging.info(f"Misclassification loss (original={target_class_id} → target={target_misclassify_id}): {misclass_loss:.4f}")
    else:
        logging.warning("Optimization did not find a best patch.")

def process_universal_patch(
    G_frozen, yolov8_model, image_file_list, bbox_folder, output_dir,
    num_generations, population_size
):
    """
    Run universal patch optimization for a whole dataset.
    Saves the best patch, its mask, and top-5 most affected images.
    """
    real_sign_images_dataset = []
    present_class_ids = set()

    for img_path in image_file_list:
        try:
            # logging.info(f"Processing image: {img_path}")  # Progress log
            bbox_yolo = get_bbox_for_image(img_path, bbox_folder)
            img_np = cv2.imread(img_path)
            if img_np is None:
                logging.warning(f"Skipping {img_path}: image could not be read")
                continue
            img_h, img_w = img_np.shape[:2]
            x1, y1, x2, y2 = yolo_bbox_to_pixel(bbox_yolo, img_w, img_h)
            patch_width = int(x2 - x1)
            patch_height = int(y2 - y1)
            if patch_width <= 0 or patch_height <= 0:
                logging.warning(f"Skipping {img_path}: invalid bbox size {x1, y1, x2, y2}")
                continue
            real_sign_images_dataset.append((img_np, (x1, y1, x2, y2), img_path))
            present_class_ids.update(get_classes_for_image(img_path, bbox_folder=bbox_folder))
        except Exception as e:
            logging.warning(f"Skipping {img_path}: {e}")
            continue
    present_class_ids = list(present_class_ids)

    if not real_sign_images_dataset:
        logging.error("No valid images with bboxes found. Exiting.")
        return None, None

    # --- Optimize a single patch for all images ---
    best_latent_code, best_loss = run_es_optimization(
        G_frozen, yolov8_model, real_sign_images_dataset, present_class_ids,
        num_generations, population_size, output_dir="checkpoints", checkpoint_interval=10
    )

    # --- Generate and save the best patch and mask ---
    with torch.no_grad():
        best_patch_tensor = G_frozen(best_latent_code, c=None, truncation_psi=0.7)
    best_patch_np = (best_patch_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    best_mask_np = extract_graffiti_mask_from_patch(best_patch_np)
    best_rgba_patch_np = extract_graffiti_rgba_from_patch(best_patch_np, best_mask_np)

    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(best_patch_np).save(os.path.join(output_dir, "best_graffiti_patch.png"))
    Image.fromarray(best_rgba_patch_np).save(os.path.join(output_dir, "best_graffiti_patch_rgba.png"))
    Image.fromarray(best_mask_np).save(os.path.join(output_dir, "best_graffiti_mask.png"))
    torch.save(best_latent_code, os.path.join(output_dir, "best_latent_code.pt"))

    # --- Evaluate and save images with largest detection drop ---
    detection_diffs = []
    for img_np, bbox, img_path in real_sign_images_dataset:
        logging.info(f"Evaluating patch on image: {img_path}")  # Progress log
        conf_before = evaluate_patch_effectiveness_multi(yolov8_model, img_np, present_class_ids)
        patched_img = realistic_patch_applier(best_rgba_patch_np, img_np, bbox)
        conf_after = evaluate_patch_effectiveness_multi(yolov8_model, patched_img, present_class_ids)
        logging.info(
            f"{img_path}: Conf before patch: {conf_before:.4f}, after patch: {conf_after:.4f}, drop: {conf_before - conf_after:.4f}"
        )
        detection_diffs.append((conf_before - conf_after, img_path, img_np, patched_img, conf_before, conf_after))

    detection_diffs.sort(reverse=True, key=lambda x: x[0])
    for i, (diff, img_path, orig_img, patched_img, conf_before, conf_after) in enumerate(detection_diffs[:5]):
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_original.png"), orig_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_patched.png"), patched_img)
        with open(os.path.join(output_dir, f"{base}_conf.txt"), "w") as f:
            f.write(f"Before: {conf_before}\nAfter: {conf_after}\nDrop: {diff}\n")

    return best_latent_code, best_loss