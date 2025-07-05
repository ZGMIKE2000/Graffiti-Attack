import os
import logging
import cv2
from PIL import Image
import torch
from patch import extract_graffiti_mask_from_patch, realistic_patch_applier
from utils import yolo_bbox_to_pixel
from optimization import run_es_optimization, evaluate_patch_effectiveness, evaluate_patch_misclassification
from numpy import np

def process_image(
    G_frozen, yolov8_model, image_path, bbox_yolo, target_class_id, target_misclassify_id,
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
        target_class_id, num_generations=num_generations, population_size=population_size
    )

    if best_latent_code is not None:
        logging.info(f"Optimization finished. Best Loss: {final_best_loss:.4f}")
        with torch.no_grad():
            best_patch_tensor = G_frozen(best_latent_code, c=None, truncation_psi=0.7)
        best_patch_np = (best_patch_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        best_mask_np = extract_graffiti_mask_from_patch(best_patch_np)
        cv2.imwrite(f"{os.path.splitext(os.path.basename(image_path))[0]}_before.png", sign_image_np)

        final_patched_image_np = realistic_patch_applier(
            best_patch_np, best_mask_np, sign_image_np, target_bbox_on_sign
        )

        # Save images
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_before.png"), sign_image_np)
        Image.fromarray(cv2.cvtColor(final_patched_image_np, cv2.COLOR_BGR2RGB)).save(
            os.path.join(output_dir, f"{base_name}_final_adversarial_patched_sign.png")
        )
        Image.fromarray(cv2.cvtColor(best_patch_np, cv2.COLOR_BGR2RGB)).save(
            os.path.join(output_dir, f"{base_name}_best_graffiti_patch.png")
        )
        logging.info("Best patch and final patched sign saved.")

        # Re-evaluate YOLOv8 on the final image
        logging.info("Re-evaluating final patched sign with YOLOv8:")
        final_results = yolov8_model(final_patched_image_np, verbose=True)
        logging.info(final_results)
        final_loss = evaluate_patch_effectiveness(yolov8_model, final_patched_image_np, target_class_id)
        logging.info(f"Final YOLOv8 loss (confidence for target class) on patched image: {final_loss:.4f}")

        # Optionally: Evaluate misclassification
        if target_misclassify_id is not None:
            misclass_loss = evaluate_patch_misclassification(
                yolov8_model, final_patched_image_np, target_class_id, target_misclassify_id
            )
            logging.info(f"Misclassification loss (original={target_class_id} → target={target_misclassify_id}): {misclass_loss:.4f}")
    else:
        logging.warning("Optimization did not find a best patch.")