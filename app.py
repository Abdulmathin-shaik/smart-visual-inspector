import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO('best.pt')

def detect_defects(img):
    try:
        # Save the input image temporarily since YOLO works better with file paths
        input_path = "temp_input.jpg"
        if isinstance(img, Image.Image):
            img.save(input_path)
        else:
            Image.fromarray(img).save(input_path)
        
        # Run prediction using file path
        results = model.predict(source=input_path)
        
        # Use plot() to draw boxes
        result_img = results[0].plot()
        
        # Add debug information
        print(f"Number of detections: {len(results[0].boxes)}")
        print(f"Boxes coordinates: {results[0].boxes.xyxy}")
        
        return Image.fromarray(result_img)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return img
demo = gr.Interface(
    fn=detect_defects,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Detected Defects"),
    title="Defect Detection with YOLO",
    description="Upload an image to detect defects using YOLO model",
    theme="default"
)
demo.launch(share=True)