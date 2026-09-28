import gradio as gr
import torch
import cv2
import numpy as np
import json
import segmentation_models_pytorch as smp

# --- Load Metrics ---
metrics_text = "### Model Performance Matrix\n"
try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    for k, v in metrics.items():
        metrics_text += f"- **{k.capitalize()}**: {v:.4f}\n"
except FileNotFoundError:
    metrics_text += "_Training metrics not found. Please train the model first._"

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None, 
        in_channels=1,
        classes=1,
    )
    # Using the colab weights file
    model.load_state_dict(torch.load('unet_dagm_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    model_loaded = True
except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")

# --- Inference Function ---
def detect_scratch(image):
    if not model_loaded:
        return None, None
    
    if image is None:
        return None, None
        
    # Old weights expect 1-channel Grayscale images
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Preprocess (1, H, W)
    input_tensor = torch.from_numpy(gray_image).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension (1, 1, H, W)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        preds = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # Threshold predictions
    mask = (preds > 0.5).astype(np.uint8) * 255
    
    # Visualization: overlay mask
    overlay = image.copy()
    
    # Make a red mask
    colored_mask = np.zeros_like(overlay)
    colored_mask[:, :, 0] = mask # Red channel
    
    # Combine
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return mask, overlay

# --- Gradio UI ---
with gr.Blocks(title="Automated Surface Scratch Detection") as demo:
    gr.Markdown("# Automated Surface Scratch and Defect Detection")
    gr.Markdown("*Industrial Quality Control Term Project - DAGM 2007 Dataset*")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(metrics_text)
            
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image (DAGM Dataset)", type="numpy")
            btn = gr.Button("Detect Defects", variant="primary")
            
        with gr.Column():
            output_mask = gr.Image(label="Predicted Defect Mask")
            output_overlay = gr.Image(label="Overlay")
            
    btn.click(fn=detect_scratch, inputs=input_image, outputs=[output_mask, output_overlay])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
