import optuna
import numpy as np

def optimize_patch_placement(dataset, graffiti_rgba_patch_np, yolov8_model, target_class_id):
    def objective(trial, image_data):
        alpha = trial.suggest_float("alpha", 0.5, 1.0)
        patch_scale = trial.suggest_float("patch_scale", 0.1, 1.0)
        x_offset = trial.suggest_float("x_offset", 0.0, 1.0)
        y_offset = trial.suggest_float("y_offset", 0.0, 1.0)

        img_np, target_bbox_on_sign, baseline_conf = image_data
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = target_bbox_on_sign
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min
        patch_x_min = int(bbox_x_min + x_offset * bbox_width)
        patch_y_min = int(bbox_y_min + y_offset * bbox_height)

        patched_image = realistic_patch_applier(
            graffiti_rgba_patch_np,
            img_np,
            target_bbox_on_sign,
            alpha=alpha,
            patch_scale=patch_scale,
            position_mode="custom",
            custom_position=(patch_x_min, patch_y_min),
            use_seamless_clone=False,
            add_blur=True,
            add_noise=True
        )

        patched_conf = evaluate_patch_effectiveness_single(yolov8_model, patched_image, target_class_id)
        loss = patched_conf - baseline_conf

        return loss

    optimized_params = {}

    for idx, (img_np, bboxes, baseline_conf) in enumerate(dataset):
        for bbox in bboxes:
            image_data = (img_np, bbox, baseline_conf)

            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, image_data), n_trials=50)

            optimized_params[idx] = {
                "bbox": bbox,
                "best_params": study.best_params
            }

            print(f"Image {idx}, BBox {bbox}: Best Parameters: {study.best_params}")

    return optimized_params