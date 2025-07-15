# Graffiti Attack

## Adversarial Patch Optimization Using StyleGAN3, YOLOv8, and Nevergrad

This project investigates the use of generative models and black-box optimization to produce naturalistic adversarial patches in the form of graffiti, aimed at degrading the performance of object detection systems in autonomous driving scenarios.

### Generator: StyleGAN3 (Fine-tuned on Graffiti Data)

The generator is based on **StyleGAN3**, fine-tuned on a curated dataset of approximately 1,000 graffiti images. StyleGAN3 is selected for:

- **High style diversity**, enabling generation of visually diverse graffiti patterns that mimic real-world variability.
- **Adaptive Data Augmentation (ADA)**, improving training robustness under limited data conditions.

### Patch Extraction and Application

Generated graffiti samples are post-processed using **OpenCV** to extract the relevant patch region, which is then overlaid onto traffic sign images.  
Currently, patches are applied at the center of the sign area for controlled testing.

- **Planned:** Integrate a placement optimization module to allow spatial variation and increase adversarial strength.

### Adversarial Optimization Strategy

The target detection model is **YOLOv8m**, treated as a **black-box**.  
Since gradient access is unavailable, **Nevergrad** (evolutionary optimizer) is used to search the StyleGAN3 latent space for graffiti samples that maximize the adversarial effect.

Two attack objectives are considered:

- **Detection Attack:** Minimize detection confidence or suppress detection entirely.
- **Classification Attack:** Induce misclassification (e.g., "Stop Sign" → "Speed Limit").

### Planned Enhancements

- **Surrogate Model Integration:** A white-box model will be embedded in GAN training to guide generation and improve transferability.
- **Cross-Model Evaluation:** Additional detectors will be tested to assess generalization of the adversarial patches.

---

## Project Structure

```
graffiti_attack/
├──src
    ├── models.py          # Loads StyleGAN3 and YOLOv8 models
    ├── patch.py           # Patch creation, masking, and application logic
    ├── optimization.py    # Evolution strategy and loss functions
    ├── utils.py           # Helper utilities
    ├── main.py            # Main experiment loop
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
