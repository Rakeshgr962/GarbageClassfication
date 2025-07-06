# 🗑️ Garbage Classification using Transfer Learning (EfficientNetB0)

This project implements a deep learning-based image classifier for garbage segregation using **EfficientNetB0** and **transfer learning**. The model can classify waste into six categories: **cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**.

---

## 📌 Features

- ✅ Trained using **EfficientNetB0** with transfer learning  
- ✅ Achieved **89.24% accuracy** on test dataset  
- ✅ Evaluated using accuracy/loss plots, confusion matrix, and classification report  
- ✅ Built a **Gradio web app** for real-time image classification  
- ✅ Optimized for speed, simplicity, and deployment readiness  

---

## 🧠 Classes

- `cardboard`  
- `glass`  
- `metal`  
- `paper`  
- `plastic`  
- `trash`

---

## 🔧 Tools & Technologies

- Python 3.10  
- TensorFlow / Keras  
- NumPy, PIL, Pandas  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Gradio (for web deployment)

---

## 📁 Folder Structure

```
📦 garbage-classification/
├── app.py                 # Gradio web app
├── EfficientNetB0.keras   # Trained model weights
├── garbage_classifier.ipynb  # Main training notebook
├── README.md              # Project description
└── dataset/               # (Optional) Folder for input data
```

---

## 🚀 Running the Project

### 1. 🧪 Train the Model (Optional)
If you'd like to retrain the model:

```bash
jupyter notebook garbage_classifier.ipynb
```

Or simply use the pre-trained `EfficientNetB0.keras`.

---

### 2. 🌐 Launch the Web App

Install Gradio:

```bash
pip install gradio
```

Run the app:

```bash
python app.py
```

Visit `http://127.0.0.1:7860` in your browser.

---

## 📊 Evaluation Metrics

- **Test Accuracy**: `89.24%`
- **Confusion Matrix**: Shows good class-wise prediction balance
- **Validation Curves**: Accuracy and loss trends monitored

---

## 📦 Model

- The trained model is saved as:  
  `EfficientNetB0.keras`

- You can load it with:
```python
from tensorflow.keras.models import load_model
model = load_model("EfficientNetB0.keras")
```

---

## 📝 Future Improvements

- Collect more balanced training data  
- Improve accuracy with advanced augmentation  
- Deploy on embedded devices or mobile  
- Integrate into smart bins or sorting systems  

---

## 🙌 Acknowledgements

- Inspired by TrashNet & Garbage dataset projects  
- Built during internship for academic purpose  
- Powered by TensorFlow, Keras, and Gradio

---

## 📎 License

This project is for academic and learning purposes. You’re free to use, modify, and share with credit.
