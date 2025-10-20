import nevergrad as ng
import numpy as np
import time
import os
import json
from utils import save_checkpoint, load_image
from models import GraffitiGenerator, ObjectDetector

class BlackBoxOptimizer:
    """
    Optimize non-gradient parameters (scale, alpha, x_rel, y_rel) using Nevergrad.
    Supports single-image or multi-image evaluation (transferability).
    """
    def __init__(self, generator: GraffitiGenerator, detector: ObjectDetector,
                 image_paths, bbox_paths, target_class_id,
                 num_generations=100, population_size=10,
                 checkpoint_dir="checkpoints", output_dir="output",
                 device='cpu'):
        self.gen = generator
        self.det = detector
        # accept single path or list
        self.image_paths = image_paths if isinstance(image_paths, (list,tuple)) else [image_paths]
        self.bbox_paths = bbox_paths if isinstance(bbox_paths, (list,tuple)) else [bbox_paths]
        self.target_class_id = target_class_id
        self.num_generations = num_generations
        self.population_size = population_size
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # define nevergrad param space
        self.parametrization = ng.p.Instrumentation(
            scale=ng.p.Scalar(lower=0.3, upper=1.2).set_mutation(sigma=0.1),
            alpha=ng.p.Scalar(lower=0.0, upper=1.0),
            x_rel=ng.p.Scalar(lower=-0.5, upper=0.5),
            y_rel=ng.p.Scalar(lower=-0.5, upper=0.5),
        )
        # use CMA as example (works well for continuous)
        self.optimizer = ng.optimizers.CMA(self.parametrization, budget=self.num_generations * self.population_size, num_workers=self.population_size)

    def _evaluate_params(self, args):
        """
        args is a tuple: (scale, alpha, x_rel, y_rel)
        Apply patch to all images/bboxes and return mean ensemble confidence for target class.
        We return loss (to minimize): mean_confidence (we want to minimize detection confidence)
        """
        scale = args["scale"]; alpha = args["alpha"]; x_rel = args["x_rel"]; y_rel = args["y_rel"]
        scores = []
        for img_path, bbox_path in zip(self.image_paths, self.bbox_paths):
            img = load_image(img_path)
            # convert bbox txt to absolute coords if bbox path provided else use precomputed best
            if bbox_path:
                from utils import bbox_from_yolo_txt
                bbox = bbox_from_yolo_txt(bbox_path, img_shape=img.shape)
            else:
                # fallback: center bbox (small)
                h,w = img.shape[:2]; bbox = (w//4, h//4, 3*w//4, 3*h//4)
            patched = self.gen.apply_patch(img, bbox, scale=scale, alpha=alpha, x_rel=x_rel, y_rel=y_rel)
            score = self.det.evaluate_target_confidence(patched, self.target_class_id)
            scores.append(score)
        mean_conf = float(np.mean(scores))
        return mean_conf  # we want to minimize this

    def run(self):
        best = None
        start = time.time()
        for gen in range(self.num_generations):
            candidates = [self.optimizer.ask() for _ in range(self.population_size)]
            losses = []
            for cand in candidates:
                vals = cand.value
                # cand.value is dict because of Instrumentation
                loss = self._evaluate_params(vals)
                # nevergrad minimizes; loss = mean_conf
                cand.loss = loss
                losses.append((cand, loss))
            # tell all
            for cand, loss in losses:
                self.optimizer.tell(cand, loss)
            # record best
            recommended = self.optimizer.provide_recommendation()
            rec_vals = recommended.value
            rec_loss = self._evaluate_params(rec_vals)
            best = {"gen": gen, "params": rec_vals, "loss": rec_loss}
            # checkpoint
            save_checkpoint(best, os.path.join(self.checkpoint_dir, f"best_gen_{gen}.json"))
            print(f"[gen {gen}] rec_loss={rec_loss:.6f} params={rec_vals}")
        total = time.time() - start
        print("Optimization finished, time(s):", total)
        # final save
        save_checkpoint(best, os.path.join(self.output_dir, "best_final.json"))
        return best