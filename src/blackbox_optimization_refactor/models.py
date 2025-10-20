import cv2
import numpy as np
from ultralytics import YOLO

class GraffitiGenerator:
    """
    Load a static patch image (RGBA or RGB+alpha) and provide apply_patch(...) to overlay
    onto a target image given bbox, scale, alpha and offsets (relative).
    Non-gradient, realistic overlay.
    """
    def __init__(self, patch_path=None, patch_img=None, use_alpha=True):
        if patch_img is None and patch_path is None:
            raise ValueError("Either patch_path or patch_img must be provided")
        if patch_img is not None:
            self.patch = patch_img.copy()
        else:
            self.patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)  # keep alpha if present
        # Ensure RGBA
        if self.patch.ndim == 2:
            self.patch = cv2.cvtColor(self.patch, cv2.COLOR_GRAY2BGRA)
        if self.patch.shape[2] == 3:
            if use_alpha:
                alpha = np.ones((self.patch.shape[0], self.patch.shape[1], 1), dtype=self.patch.dtype) * 255
                self.patch = np.concatenate([self.patch, alpha], axis=2)
        self.use_alpha = use_alpha

    def get_mask(self):
        # Alpha channel normalized 0..1
        alpha = self.patch[:, :, 3].astype(np.float32) / 255.0
        return alpha

    def apply_patch(self, image, bbox_xyxy, scale=0.7, alpha=0.9, x_rel=0.0, y_rel=0.0, interp=cv2.INTER_CUBIC):
        """
        image: HxWx3 BGR numpy array
        bbox_xyxy: (x1,y1,x2,y2) absolute coords in image (pixels)
        scale: patch size relative to bbox (0..2)
        alpha: global blend (0..1)
        x_rel,y_rel: offset in bbox widths/heights units
        Returns composited image (BGR uint8) - guaranteed to be inside bbox.
        """
        img = image.copy()
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        # clamp bbox to image
        x1 = max(0, min(iw-1, x1)); x2 = max(1, min(iw, x2))
        y1 = max(0, min(ih-1, y1)); y2 = max(1, min(ih, y2))
        bw = max(1, x2 - x1); bh = max(1, y2 - y1)

        # compute target size and ensure it does not exceed bbox
        target_w = max(1, int(round(bw * float(scale))))
        target_h = max(1, int(round(bh * float(scale))))
        target_w = min(target_w, bw)
        target_h = min(target_h, bh)

        # Resize BGRA patch to target (keep alpha)
        # self.patch is expected BGRA
        patch_resized = cv2.resize(self.patch, (target_w, target_h), interpolation=interp)
        if patch_resized.shape[2] == 4:
            patch_rgb = patch_resized[..., :3].astype(np.float32)
            mask = (patch_resized[..., 3].astype(np.float32) / 255.0)[..., None]
        else:
            patch_rgb = patch_resized[..., :3].astype(np.float32)
            mask = np.ones((target_h, target_w, 1), dtype=np.float32)

        # apply global alpha multipler and clamp 0..1
        mask = np.clip(mask * float(alpha), 0.0, 1.0)

        # compute center + offset and top-left, then clamp inside bbox
        cx = x1 + bw/2.0 + float(x_rel) * bw
        cy = y1 + bh/2.0 + float(y_rel) * bh
        px1 = int(round(cx - target_w/2.0)); py1 = int(round(cy - target_h/2.0))
        # clamp so patch stays inside bbox
        px1 = int(np.clip(px1, x1, x2 - target_w))
        py1 = int(np.clip(py1, y1, y2 - target_h))
        px2 = px1 + target_w; py2 = py1 + target_h

        # Clip to image borders (safety)
        sx1 = max(0, -px1); sy1 = max(0, -py1)
        dx1 = max(0, px1); dy1 = max(0, py1)
        sx2 = target_w - max(0, px2 - iw); sy2 = target_h - max(0, py2 - ih)
        if dx1 >= iw or dy1 >= ih or sx2 <= sx1 or sy2 <= sy1:
            return img  # nothing to paste

        patch_crop = patch_rgb[sy1:sy2, sx1:sx2, :].astype(np.float32)
        mask_crop = mask[sy1:sy2, sx1:sx2, :].astype(np.float32)
        region = img[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1), :].astype(np.float32)

        # Blend (BGR order preserved)
        a = mask_crop
        comp = (1.0 - a) * region + a * patch_crop
        img[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1), :] = np.clip(comp, 0, 255).astype(np.uint8)
        return img
    
class ObjectDetector:
    """
    Wrap Ultralytics YOLO models for black-box evaluation.
    Accepts one or multiple checkpoints (ensemble). Provides evaluate(image, target_class_id),
    returning an aggregated score (mean confidence of best detection per model).
    """
    def __init__(self, model_paths, device='cpu'):
        if isinstance(model_paths, (list, tuple)):
            self.models = [YOLO(p) for p in model_paths]
        else:
            self.models = [YOLO(model_paths)]
        # move models to device
        for m in self.models:
            try:
                m.model.to(device)
            except Exception:
                pass
        self.device = device

    def detect(self, image, conf=0.001):
        """
        Run each model, return list of results per model (Ultralytics results objects).
        """
        results = []
        for m in self.models:
            # Ultralytics predict inference; we treat detector as black-box so use predict
            r = m(image, imgsz=640)  # keep original size or let model handle preprocessing
            results.append(r[0])
        return results

    def evaluate_target_confidence(self, image, target_class_id):
        """
        For each model, find the highest confidence for target_class_id. Return mean across models.
        """
        results = self.detect(image)
        confidences = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                confidences.append(0.0)
                continue
            # boxes.conf, boxes.cls
            cls = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            # max conf for target class
            mask = cls == target_class_id
            if mask.any():
                confidences.append(float(confs[mask].max()))
            else:
                confidences.append(0.0)
        return float(np.mean(confidences))