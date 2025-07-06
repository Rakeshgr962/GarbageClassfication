import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained EfficientNetB0 model
model = tf.keras.models.load_model("EfficientNetB0.keras")

# Class names (replace if you used different labels/order)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Define the prediction function
def classify_image(image):
    # Resize to match input shape
    image = image.resize((124, 124))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# Build Gradio UI
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Garbage Classifier (EfficientNetB0)",
    description="Upload an image of garbage. The model will classify it into 1 of 6 categories."
)

# Launch the web app
demo.launch()
