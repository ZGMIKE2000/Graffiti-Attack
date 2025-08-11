# Graffiti Attack

## Adversarial Patch Optimization Using StyleGAN3, YOLOv8, and Nevergrad

This project investigates the use of generative models and black-box optimization to produce naturalistic adversarial patches in the form of graffiti, aimed at degrading the performance of object detection systems in autonomous driving scenarios.

### Generator: StyleGAN3 (Fine-tuned on Graffiti Data)

The generator is based on **StyleGAN3**, fine-tuned on a curated dataset of approximately 1,000 graffiti images. StyleGAN3 is selected for:

- **High style diversity**, enabling generation of visually diverse graffiti patterns that mimic real-world variability.
- **Adaptive Data Augmentation (ADA)**, improving training robustness under limited data conditions.

Naturalness survey to be conducted on the generated patches.

### Patch Extraction and Application

Generated graffiti samples are post-processed using **OpenCV** to extract the relevant patch region, which is then overlaid onto traffic sign images.  
The patches are applied after some augmentations from Kornia repository and then tested against the yolov8m model. Current approach of the loss calculation is based the average loss of each patched image evaluation on the yolo model wrt the pre-patched image.

### Adversarial Optimization Strategy

The target detection model is **YOLOv8m**, treated as a **black-box**.  
Since gradient access is unavailable, **Nevergrad** (evolutionary optimizer) is used to search the StyleGAN3 latent space for graffiti samples that maximize the adversarial effect.

Two attack objectives are considered:

- **Detection Attack:** Minimize detection confidence or suppress detection entirely.
- **Classification Attack:** Induce misclassification (e.g., "Stop Sign" → "Speed Limit").

Currently focusing on the Detection Attack with an aim on Stop signs, considering it's critical role in autonomous driving scenarios.

About the Yolo model: The model is finetuned on a custom dataset retrieved from the mapillary dataset using its API.

### Planned Enhancements

- **Surrogate Model Integration:** A white-box model will be embedded in GAN training to guide generation (edges and shapes).
- **Cross-Model Evaluation:** Additional detectors will be tested to assess transferability of the adversarial patches.

### Current results summary

In prior experiments, adversarial patches smaller than 90% of the traffic sign’s area exhibited limited effectiveness against the YOLO detector. However, when patch scale, transparency (alpha), and positional coordinates are jointly optimized within the evolutionary search, successful attacks (stop signs not detected) are achievable with patches occupying as little as 50% of the sign’s area.

Experimentations are still on-going, there will be tables of results for clarity.

-Eventual testing on other classes and robust benchmarking results will be retrieved.

---

## Project Structure

```
graffiti_attack/
├──src
    ├── eot.py             # EOT augmentation pipeline from Kornia
    ├── main.py            # Main file
    ├── models.py          # Loads StyleGAN3 and YOLOv8 models
    ├── optimization.py    # Evolution strategy and loss functions
    ├── patch.py           # Patch creation, masking, and application logic
    ├── placement.py       # Helper for patching mechanism
    ├── processing.py      # Helper for optimization mechanism
    ├── utils.py           # Helper utilities
    ├── test_models.py     # Sanity check for model loading
├── requirements.txt   # Dependencies (excludes StyleGAN3)
└── README.md          # This file
```

## Notes

This work is part of my Master's thesis project conducted at Waseda University under the supervision of Prof. Tatsuya Mori.

---

## License

- This repo: [MIT License](LICENSE)
- StyleGAN3: [NVlabs License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
