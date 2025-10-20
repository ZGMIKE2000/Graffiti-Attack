import os
import cv2
import torch
import logging
import itertools
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F

from patch_overlay import overlay_patch_rgba
from ultralytics.utils.loss import v8DetectionLoss

class GraffitiTrainer:
    """
    Trainer for adversarial latent optimization using a StyleGAN3 generator and YOLOv8 detector.
    Applies a generated patch to images, computes detection loss, and optimizes the latent vector.
    """

    def __init__(self, generator, yolo, dataloader, dataset, device="cuda", output_root="output", log_file=None):
        self.output_root = output_root
        # os.makedirs(output_root, exist_ok=True)
        if log_file is None:
            log_file = os.path.join(output_root, "training.log")
        self.generator = generator
        self.yolo = yolo
        self.dataloader = dataloader
        self.dataset = dataset
        self.device = device
        self.yolo_loss = v8DetectionLoss(self.yolo.model)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GraffitiTrainer")

    def train_latent(self, z, optimizer, num_steps=10):
        """
        Optimize the latent vector z to maximize YOLOv8 detection loss on patched images.

        Args:
            z: Latent vector (torch.Tensor, requires_grad=True)
            optimizer: Optimizer for z
            num_steps: Number of optimization steps
            target_class: (Unused) Optionally focus on a specific class
            reduction: (Unused) Loss reduction method
        """
        # --- Create output directories ---
        composite_dir = os.path.join(self.output_root, "composites")
        yolo_vis_dir = os.path.join(self.output_root, "yolo_visualizations")
        grad_hist_dir = os.path.join(self.output_root, "grad_histograms")
        latent_vector_dir = os.path.join(self.output_root, "latent_vectors")
        os.makedirs(composite_dir, exist_ok=True)
        os.makedirs(yolo_vis_dir, exist_ok=True)
        os.makedirs(grad_hist_dir, exist_ok=True)
        os.makedirs(latent_vector_dir, exist_ok=True)

        loss_history = []
        grad_norm_history = []
        dataloader_iter = itertools.cycle(self.dataloader)  # Infinite iterator over batches
        class_names = self.yolo.model.names if hasattr(self.yolo.model, "names") else None

        for step in range(num_steps):
            batch = next(dataloader_iter)
        
            # --- Extract bounding boxes and original image ---
            orig_bboxes = batch.get("orig_bboxes", [])
            orig_img_tensor = batch["orig_img"].to(self.device)

            # --- Generate patch and mask ---
            patch = self.generator.generate(z)
            if torch.isnan(patch).any() or torch.isinf(patch).any():
                self.logger.warning(f"Patch is NaN or Inf at step {step+1}, skipping step.")
                continue

            mask, _ = self.generator.auto_quantile_mask(patch)
            patch = patch.clamp(0, 1)
            mask = mask.clamp(0, 1)
            patch_rgba = torch.cat([patch, mask], dim=1)

            # --- Overlay patch on all bboxes in the image ---
            composite_tensor = orig_img_tensor.clone()
            patch_scale = 0.8  # 70% of bbox size (change as needed)

            # Inside your bbox loop:
            for bbox in orig_bboxes:
                _, x1, y1, x2, y2 = bbox
                bbox_h, bbox_w = y2 - y1, x2 - x1
                # Scale patch size
                patch_h = int(bbox_h * patch_scale)
                patch_w = int(bbox_w * patch_scale)
                patch_resized = F.interpolate(patch_rgba, size=(patch_h, patch_w), mode="bicubic", align_corners=False)
                # Center the patch in the bbox
                y_offset = y1 + (bbox_h - patch_h) // 2
                x_offset = x1 + (bbox_w - patch_w) // 2
                bbox_tensor = torch.tensor([[x_offset, y_offset, x_offset + patch_w, y_offset + patch_h]], device=self.device)
                composite_tensor = overlay_patch_rgba(composite_tensor, patch_resized, bbox_tensor, alpha=1.0)

            # --- Save composite image for inspection ---
            composite_path = os.path.join(composite_dir, f"composite_step_{step+1}.png")
            if step % 100 == 0 or step == num_steps - 1:  # Save every 10 steps (or last step)
                self.dataset.save_composite(composite_tensor, composite_path)


            # --- Prepare input for YOLO ---
            composite_for_yolo = composite_tensor.clamp(0, 1).to(self.device)
            if composite_for_yolo.shape[-2:] != (640, 640):
                composite_for_yolo = F.interpolate(composite_for_yolo, size=(640, 640), mode="bilinear", align_corners=False)
            if composite_for_yolo.dim() == 3:
                composite_for_yolo = composite_for_yolo.unsqueeze(0)  # [1, 3, 640, 640]

            # --- Prepare YOLO label batch dict ---
            h0, w0 = orig_img_tensor.shape[-2:]  # original image size
            r = min(640 / h0, 640 / w0)
            new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
            dw = (640 - new_unpad[0]) / 2  # width padding
            dh = (640 - new_unpad[1]) / 2  # height padding
            class_ids, bboxes = [], []
            for bbox in orig_bboxes:
                class_id = int(bbox[0][0])
                x1, y1, x2, y2 = [float(bbox[j][0]) for j in range(1, 5)]
                # Scale
                x1 = x1 * r + dw
                x2 = x2 * r + dw
                y1 = y1 * r + dh
                y2 = y2 * r + dh
                # Normalize
                x1 /= 640
                x2 /= 640
                y1 /= 640
                y2 /= 640
                class_ids.append([class_id])
                bboxes.append([x1, y1, x2, y2])
            if len(class_ids) == 0:
                class_ids = torch.zeros((0, 1), device=self.device)
                bboxes = torch.zeros((0, 4), device=self.device)
            else:
                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.float32)
                bboxes = torch.tensor(bboxes, device=self.device, dtype=torch.float32)

            batch_dict = {
                "batch_idx": torch.zeros((class_ids.shape[0], 1), device=self.device, dtype=torch.float32),
                "cls": class_ids,
                "bboxes": bboxes,
            }

            # --- Ensure YOLOv8 loss hyperparameters are present ---
            default_hyp = {"box": 0.05, "cls": 0.5, "dfl": 1.5}
            for k, v in default_hyp.items():
                if k not in self.yolo.model.args:
                    self.yolo.model.args[k] = v

            # --- Forward through YOLO and compute loss ---
            self.yolo.model.train()
            preds = self.yolo.model(composite_for_yolo)
            loss, _ = self.yolo_loss(preds, batch_dict)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                self.logger.warning(f"Loss is NaN or Inf at step {step+1}, skipping step.")
                continue

            patch_strength = patch.abs().mean()
            z_reg_lambda = 0.01
            z_reg_loss = z.norm()
            loss = loss + 0.05 * patch_strength + z_reg_lambda * z_reg_loss  # tune 0.05 as needed

            # --- Check for NaNs ---
            if torch.isnan(loss).any() or torch.isinf(loss).any() or torch.isnan(patch).any() or torch.isinf(patch).any():
                self.logger.warning("Loss or patch is NaN or Inf, skipping backward/step.")
                continue

            # --- Backpropagation and logging ---
            loss_scalar = loss.sum().item()
            loss_history.append(loss_scalar)
            optimizer.zero_grad()

            try:
                loss.sum().backward()
            except RuntimeError as e:
                if "returned nan values" in str(e).lower() or "nan" in str(e).lower():
                    self.logger.warning(f"NaN detected in backward at step {step+1}, skipping optimizer step. Error: {e}")
                    optimizer.zero_grad()
                    continue
                else:
                    raise  # re-raise if it's a different error

            if z.grad is not None and (torch.isnan(z.grad).any() or torch.isinf(z.grad).any()):
                self.logger.warning(f"Gradient is NaN or Inf at step {step+1}, skipping optimizer step.")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_([z], max_norm=10.0)

            # Loss breakdown (box, cls, dfl)
            box_loss = loss[0].item() if loss.numel() > 0 else float('nan')
            cls_loss = loss[1].item() if loss.numel() > 1 else float('nan')
            dfl_loss = loss[2].item() if loss.numel() > 2 else float('nan')

            # Latent statistics
            z_norm = z.norm().item()
            # z_min = z.min().item()
            # z_max = z.max().item()
            # z_mean = z.mean().item()

            # Gradient statistics
            grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
            grad_norm_history.append(grad_norm)
            grad_min = z.grad.min().item() if z.grad is not None else float('nan')
            grad_max = z.grad.max().item() if z.grad is not None else float('nan')
            grad_mean = z.grad.mean().item() if z.grad is not None else float('nan')

            self.logger.info(
                f"Step {step+1:03d} | "
                f"Loss: {loss_scalar:.4f} (box: {box_loss:.4f}, cls: {cls_loss:.4f}, dfl: {dfl_loss:.4f}) | "
                f"z norm: {z_norm:.4f} | "
                f"z grad norm: {grad_norm:.4f} | z.grad min: {grad_min:.4f} | z.grad max: {grad_max:.4f} | z.grad mean: {grad_mean:.4f}"
            )

            # --- Visualize YOLO predictions on the patched image ---
            if step % 10 == 0 or step == num_steps - 1:  # Save every 10 steps (or last step)
                # Convert tensor to numpy image
                img_np = composite_for_yolo[0].detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Get YOLO predictions (use model's inference mode)
                self.yolo.model.eval()
                with torch.no_grad():
                    composite_for_yolo_clamped = composite_for_yolo.clamp(0, 1)
                    results = self.yolo(composite_for_yolo_clamped, verbose=False)
                self.yolo.model.train()

                # Draw predicted boxes
                if hasattr(results, "boxes"):
                    boxes = results.boxes
                elif isinstance(results, (list, tuple)) and hasattr(results[0], "boxes"):
                    boxes = results[0].boxes
                else:
                    boxes = None

                predicted_class_names = []
                if boxes is not None and class_names is not None:
                    for i in range(len(boxes.xyxy)):
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                        class_name = class_names[cls] if cls < len(class_names) else str(cls)
                        predicted_class_names.append(class_name)
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_np, f"{cls}:{conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Print all predicted class names in the lower right corner
                if predicted_class_names:
                    text = "Predicted: " + ", ".join(predicted_class_names)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    color = (0, 255, 0)
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x = img_np.shape[1] - text_width - 10
                    y = img_np.shape[0] - 10
                    cv2.putText(img_np, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

                yolo_vis_path = os.path.join(yolo_vis_dir, f"yolo_pred_step_{step+1}.png")
                cv2.imwrite(yolo_vis_path, img_np)

            # # --- Plot and save gradient histogram ---
            # if step % 10 == 0 or step == num_steps - 1:  # Save every 10 steps (or last step)
            #     plt.figure()
            #     plt.hist(z.grad.detach().cpu().numpy().flatten(), bins=50)
            #     plt.title(f"z.grad histogram step {step+1}")
            #     plt.xlabel("Gradient value")
            #     plt.ylabel("Frequency")
            #     plt.tight_layout()
            #     grad_hist_path = os.path.join(grad_hist_dir, f"z_grad_hist_step_{step+1}.png")
            #     plt.savefig(grad_hist_path)
            #     plt.close()

            if step % 100 == 0 or step == num_steps - 1:
                latent_path_step = os.path.join(latent_vector_dir, f"latent_step_{step+1}.pt")
                torch.save(z.detach().cpu(), latent_path_step)
                self.logger.info(f"Latent vector saved to {latent_path_step}")

            optimizer.step()
            # Clamp z after optimizer step to prevent explosion
            with torch.no_grad():
                z.clamp_(-10, 10)  # Adjust range as needed

            # Add at the end of each training loop iteration, just before the next step
            del composite_tensor, composite_for_yolo, patch, mask, patch_rgba, preds, loss
            torch.cuda.empty_cache()

        # --- Save optimized latent vector ---
        latent_path = os.path.join(self.output_root, "optimized_latent.pt")
        torch.save(z.detach().cpu(), latent_path)
        self.logger.info(f"Optimized latent vector saved to {latent_path}")

        # --- Plot loss curve ---
        plt.figure()
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Latent Optimization Loss Curve")
        plt.legend()
        plt.tight_layout()
        loss_curve_path = os.path.join(self.output_root, "latent_optimization_loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        self.logger.info(f"Loss curve saved to {loss_curve_path}")
        
        # --- moving average of loss ---
        window = 20  # or any window size you like
        loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')

        plt.figure()
        plt.plot(loss_history, label="Raw Loss", alpha=0.4)
        plt.plot(range(window-1, len(loss_history)), loss_smooth, label=f"Moving Avg (window={window})", color='red')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Latent Optimization Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_curve_path)
        plt.close()

        # --- Plot gradient curve ---
        plt.figure()
        plt.plot(grad_norm_history, label="Raw Grad Norm", alpha=0.4)

        # Moving average of gradient norm
        if len(grad_norm_history) >= window:
            grad_norm_smooth = np.convolve(grad_norm_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(grad_norm_history)), grad_norm_smooth, label=f"Moving Avg (window={window})", color='red')

        plt.xlabel("Step")
        plt.ylabel("Gradient Norm")
        plt.title("Latent Gradient Norm Curve")
        plt.legend()
        plt.tight_layout()
        grad_norm_curve_path = os.path.join(grad_hist_dir, "latent_grad_norm_curve.png")
        plt.savefig(grad_norm_curve_path)
        plt.close()
        self.logger.info(f"Gradient norm curve saved to {grad_norm_curve_path}")