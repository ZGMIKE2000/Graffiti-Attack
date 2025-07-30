import os
import logging
import cv2
from PIL import Image
import torch
from patch import extract_graffiti_mask_from_patch, extract_graffiti_rgba_from_patch, realistic_patch_applier
from utils import yolo_bbox_to_pixel
from optimization import run_es_optimization
import numpy as np

def process_image(
    G_frozen, yolov8_model, images_path, bbox_folder, target_class_id,
    num_generations, population_size, output_dir
):
    real_sign_images_dataset = []
    for image_path in images_path:
        img_np = cv2.imread(image_path)
        if img_np is None:
            logging.warning(f"Skipping {image_path}: image could not be read")
            continue
        img_h, img_w = img_np.shape[:2]
        bbox_file = os.path.join(bbox_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        bboxes = []
        with open(bbox_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                bbox_yolo = list(map(float, parts[1:5]))
                if cls_id == target_class_id:
                    x1, y1, x2, y2 = yolo_bbox_to_pixel(bbox_yolo, img_w, img_h)
                    bboxes.append((x1, y1, x2, y2))
        if bboxes:
            real_sign_images_dataset.append((img_np, bboxes, image_path))

    logging.info(f"Loaded {len(real_sign_images_dataset)} images for optimization.")

    best_latent_code, _ = run_es_optimization(
        G_frozen, yolov8_model, real_sign_images_dataset,
        target_class_id,
        num_generations=num_generations, population_size=population_size
    )

    logging.info("Black-Box Patch Optimization completed.")

    if best_latent_code is not None:
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            best_patch_tensor = G_frozen(best_latent_code, c=None, truncation_psi=0.7)
        best_patch_np = (best_patch_tensor[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        best_mask_np = extract_graffiti_mask_from_patch(best_patch_np)
        best_rgba_patch_np = extract_graffiti_rgba_from_patch(best_patch_np, best_mask_np)

        Image.fromarray(best_rgba_patch_np).save(os.path.join(output_dir, "best_graffiti_patch_rgba.png"))
        Image.fromarray(best_mask_np).save(os.path.join(output_dir, "best_graffiti_mask.png"))
        Image.fromarray(best_patch_np).save(os.path.join(output_dir, "best_graffiti_patch.png"))
        torch.save(best_latent_code, os.path.join(output_dir, "best_latent_code.pt"))

        for img_np, bboxes, image_path in real_sign_images_dataset:
            patched_img = img_np.copy()
            for bbox in bboxes:
                patched_img = realistic_patch_applier(
                    best_rgba_patch_np,
                    patched_img,
                    bbox,
                    alpha=0.8,
                    patch_scale=0.5,
                    position_mode='center',
                    use_seamless_clone=None,
                    add_blur=True,
                    add_noise=True
                )
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_patched.png"), patched_img)