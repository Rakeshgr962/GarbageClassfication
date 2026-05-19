# Garbage Classification using Transfer Learning ♻️

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Rakeshgr962/GarbageClassfication?style=flat-square)](https://github.com/Rakeshgr962/GarbageClassfication)

---

## What it does

This project is an image classification system that looks at a photo of waste and tells you exactly what kind of garbage it is: **cardboard, glass, metal, paper, plastic, or general trash**.

The model is built on top of EfficientNetB0 (pre-trained on ImageNet) using transfer learning — meaning instead of training from scratch, we fine-tuned a model that already understands visual features, then taught it to recognize waste categories. It ships with a Gradio web interface so anyone can upload a photo and get a classification instantly, no code required.

---

## Why it matters

Waste misclassification is a real problem at scale. When people throw plastic into cardboard bins or glass into general trash, it contaminates entire recycling batches and sends recoverable material to landfill. An automated classifier like this can be integrated into smart bin systems, conveyor-belt sorting lines, or mobile apps to catch misclassification at the point of disposal — not after contamination has already happened.

The internship this was built for (Edunet Foundation / Shell India / AICTE) specifically targeted this problem, and the model achieved **94% inference accuracy** on a custom dataset of 5,500+ images after pipeline optimization.

---

## Model details

| Parameter | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Training strategy | Transfer Learning (ImageNet weights) |
| Test accuracy | **89.24%** (public dataset) / **94%** (internship dataset) |
| Output classes | Cardboard, Glass, Metal, Paper, Plastic, Trash |
| Input size | 124 × 124 px |
| Regularization | Dropout (0.2), Global Average Pooling |
| Optimizer | Adam |

---

## How to use it

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Gradio, NumPy, Pillow

### Install dependencies

```bash
git clone https://github.com/Rakeshgr962/GarbageClassfication.git
cd GarbageClassfication
pip install tensorflow gradio numpy Pillow
```

### Run the web interface

```bash
python app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`). Drop in any image of waste and the model returns a ranked probability for each of the 6 categories.

### Use the model directly in Python

```python
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("EfficientNetB0.keras")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img = Image.open("your_image.jpg").resize((124, 124))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

predictions = model.predict(img_array)[0]
top_class = class_names[np.argmax(predictions)]
confidence = predictions.max() * 100

print(f"{top_class} ({confidence:.1f}% confidence)")
```

---

## Project structure

```
GarbageClassfication/
├── app.py              # Gradio web interface — run this to use the model
├── FinalSave.ipynb     # Full training notebook: data loading, augmentation,
│                       # model fine-tuning, evaluation, and confusion matrix
├── EfficientNetB0.keras # Saved trained model weights
└── README.md
```

---

## Retraining the model

Open `FinalSave.ipynb` in Jupyter. The notebook walks through:
1. Loading and splitting the dataset
2. Applying augmentation (horizontal flip, zoom, rotation)
3. Building the EfficientNetB0 transfer learning head
4. Training with early stopping
5. Evaluating with accuracy/loss curves and a full confusion matrix

---

## License

MIT — see [LICENSE](LICENSE).
