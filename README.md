# Garbage Classification using EfficientNetB0 ♻️

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg?style=flat-square)](https://tensorflow.org)
[![License](https://img.shields.io/github/license/Rakeshgr962/GarbageClassfication?style=flat-square)](LICENSE)

An AI-driven image classification system that automates waste sorting to support sustainable recycling. Built using **EfficientNetB0** transfer learning, the model classifies images of waste into six discrete categories: cardboard, glass, metal, paper, plastic, and trash. 

---

## 📈 Model Performance
- **Base Architecture**: EfficientNetB0 (Pre-trained on ImageNet)
- **Validation Accuracy**: **89.24%**
- **Classes**: `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`
- **Techniques Used**: Global Average Pooling, Dropout regularization (0.2), Adam Optimizer.

---

## 🛠️ Tech Stack
- **Model Training**: Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn
- **Development Environment**: Jupyter Notebooks
- **Interactive UI**: Gradio

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- PIL (Pillow) & Gradio

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rakeshgr962/GarbageClassfication.git
   cd GarbageClassfication
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow gradio numpy Pillow
   ```

3. **Launch the Gradio Interface**:
   ```bash
   python app.py
   ```
   *Open the generated local URL (usually `http://127.0.0.1:7860`) in your web browser to upload images and classify waste.*

---

## 📂 Project Structure
```
GarbageClassfication/
├── app.py                   # Gradio interface web server
├── FinalSave.ipynb          # Jupyter Notebook containing training and evaluation
├── EfficientNetB0.keras     # Trained TensorFlow model weight file
└── README.md                # Project documentation
```

---

## 📄 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
