import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Geometric-Shapes-Classification"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping with symbols
labels = {
    "0": "Circle ◯",
    "1": "Kite ⬰",
    "2": "Parallelogram ▰",
    "3": "Rectangle ▭",
    "4": "Rhombus ◆",
    "5": "Square ◼",
    "6": "Trapezoid ⏢",
    "7": "Triangle ▲"
}

def classify_shape(image):
    """Classifies the geometric shape in the input image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_shape,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Geometric Shapes Classification",
    description="Upload an image to classify geometric shapes such as circle, triangle, square, and more."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
