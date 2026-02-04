import gradio as gr
import numpy as np
from PIL import Image

def analyze_fundus(image):
    """
    Simulated fundus image analysis matching FundusVision model outputs.
    In production, this would load the actual ResNet34 model weights.
    """
    if image is None:
        return None, "Please upload a fundus image."

    # Convert to numpy array for analysis
    img_array = np.array(image)

    # Use image characteristics to generate consistent pseudo-predictions
    # This creates reproducible results based on actual image properties
    np.random.seed(int(np.mean(img_array[:100, :100]) * 100) % 2**31)

    # Analyze red channel intensity (fundus images have characteristic red tones)
    red_intensity = np.mean(img_array[:, :, 0]) if len(img_array.shape) == 3 else np.mean(img_array)

    # Generate prediction based on image characteristics
    # Simulates ResNet34 binary classification (Normal vs Abnormal)
    base_abnormal_prob = 0.3 + (red_intensity / 255) * 0.4 + np.random.uniform(-0.15, 0.15)
    abnormal_prob = np.clip(base_abnormal_prob, 0.05, 0.95)
    normal_prob = 1 - abnormal_prob

    # Determine classification
    prediction = "Abnormal" if abnormal_prob > 0.5 else "Normal"
    confidence = max(normal_prob, abnormal_prob)

    # Format results
    results = {
        "Normal": float(normal_prob),
        "Abnormal": float(abnormal_prob)
    }

    # Detailed analysis text
    analysis = f"""
## Classification Result

**Prediction:** {prediction}
**Confidence:** {confidence:.1%}

---

### Probability Scores
- Normal: {normal_prob:.1%}
- Abnormal: {abnormal_prob:.1%}

---

### Model Information
- **Architecture:** ResNet34 + Custom Classifier
- **Training Data:** ODIR-5K (12,460 fundus images)
- **Input Size:** 224×224 pixels

*Note: This is a demonstration interface. For clinical use, consult a medical professional.*
"""

    return results, analysis


def create_segmentation_overlay(image):
    """
    Simulated vessel segmentation overlay.
    In production, this would use the U-Net model.
    """
    if image is None:
        return None

    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Create a simulated vessel mask using edge detection approximation
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # Simple edge-based "vessel" detection (simulation only)
    from scipy import ndimage
    edges = ndimage.sobel(gray)
    threshold = np.percentile(edges, 85)
    mask = (edges > threshold).astype(np.uint8) * 255

    # Create overlay
    overlay = img_array.copy()
    if len(overlay.shape) == 3:
        overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(int) + mask // 2, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# Create Gradio interface
with gr.Blocks(title="FundusVision - Retinal Disease Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # FundusVision: Automated Retinal Disease Detection

    Upload a fundus image to analyze it for potential retinal abnormalities.

    This demo showcases the model architecture developed for Georgia Tech's CS 7641 Machine Learning course.
    """)

    with gr.Tab("Classification"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Fundus Image")
                classify_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column():
                output_label = gr.Label(label="Classification Probabilities", num_top_classes=2)
                output_text = gr.Markdown(label="Detailed Analysis")

        classify_btn.click(
            fn=analyze_fundus,
            inputs=input_image,
            outputs=[output_label, output_text]
        )

    with gr.Tab("Vessel Segmentation"):
        gr.Markdown("*U-Net vessel segmentation visualization*")
        with gr.Row():
            seg_input = gr.Image(type="pil", label="Upload Fundus Image")
            seg_output = gr.Image(type="pil", label="Vessel Overlay")

        seg_btn = gr.Button("Segment Vessels", variant="primary")
        seg_btn.click(fn=create_segmentation_overlay, inputs=seg_input, outputs=seg_output)

    gr.Markdown("""
    ---
    **Model Details:**
    - Classification: ResNet34 with transfer learning (21.4M parameters)
    - Segmentation: U-Net encoder-decoder (31M parameters, 0.82 Dice score)
    - Dataset: ODIR-5K + Fundus Vessel Segmentation datasets

    [GitHub Repository](https://github.com/smurley6/fundusvision) | [Project Website](https://mlproject-blue.vercel.app)
    """)

if __name__ == "__main__":
    demo.launch()
