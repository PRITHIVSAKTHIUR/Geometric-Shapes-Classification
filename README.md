![zdfdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/z98n2b4kPFRoxPGKddrno.png)
  
# **Geometric-Shapes-Classification**

> **Geometric-Shapes-Classification** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a multi-class shape recognition task. It classifies various geometric shapes using the **SiglipForImageClassification** architecture.

```py
Classification Report:
                 precision    recall  f1-score   support

       Circle ◯     0.9921    0.9987    0.9953      1500
         Kite ⬰     0.9927    0.9927    0.9927      1500
Parallelogram ▰     0.9926    0.9840    0.9883      1500
    Rectangle ▭     0.9993    0.9913    0.9953      1500
      Rhombus ◆     0.9846    0.9820    0.9833      1500
       Square ◼     0.9914    0.9987    0.9950      1500
    Trapezoid ⏢     0.9966    0.9793    0.9879      1500
     Triangle ▲     0.9772    0.9993    0.9881      1500

       accuracy                         0.9908     12000
      macro avg     0.9908    0.9908    0.9907     12000
   weighted avg     0.9908    0.9908    0.9907     12000
```

![download (3).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/WAdeb9cy5DBb-zZ0TPx0k.png)

The model categorizes images into the following classes:

- **Class 0:** Circle ◯  
- **Class 1:** Kite ⬰  
- **Class 2:** Parallelogram ▰  
- **Class 3:** Rectangle ▭  
- **Class 4:** Rhombus ◆  
- **Class 5:** Square ◼  
- **Class 6:** Trapezoid ⏢  
- **Class 7:** Triangle ▲  

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

The **Geometric-Shapes-Classification** model is designed to recognize basic geometric shapes in images. Example use cases:

- **Educational Tools:** For learning and teaching geometry visually.  
- **Computer Vision Projects:** As a shape detector in robotics or automation.  
- **Image Analysis:** Recognizing symbols in diagrams or engineering drafts.  
- **Assistive Technology:** Supporting shape identification for visually impaired applications.
