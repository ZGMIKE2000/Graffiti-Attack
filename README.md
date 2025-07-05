# 🎨 Graffiti Attack

Adversarial patch optimization using **StyleGAN3**, **YOLOv8**, and **Nevergrad** to generate natural-looking graffiti attacks on object detection systems.

---

## 📁 Project Structure

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

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/graffiti_attack.git
cd graffiti_attack
```

### 2. Set Up Python Environment

Install PyTorch for your system manually:  
👉 https://pytorch.org/get-started/locally/

Then install other requirements:

```bash
pip install -r requirements.txt
```

### 3. Install StyleGAN3 (for `dnnlib`, `legacy`)

StyleGAN3 does **not** include a `setup.py`. Do **not** copy it into your repo.

Instead:

```bash
git clone https://github.com/NVlabs/stylegan3.git ~/stylegan3
```

Then in your scripts (e.g., `models.py`), add:

```python
import sys
sys.path.append('/home/yourusername/stylegan3')  # Adjust path if needed
```

---

## 🚀 Usage

Edit `main.py` to set:
- dataset path
- model checkpoint paths
- optimization settings

Then run:

```bash
python main.py
```

---

## 🧪 Testing

To check if model loading works, update paths in `test_models.py`:

```bash
python test_models.py
```

---

## 📝 Notes

If you get errors like `ModuleNotFoundError: No module named 'click'`, fix them with:

```bash
conda install click
# or
pip install click
```

---

## 📚 License

- This repo: [Your License Here]  
- StyleGAN3: [NVlabs License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
