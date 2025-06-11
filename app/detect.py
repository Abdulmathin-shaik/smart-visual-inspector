from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

model = YOLO('/Users/abdulshaik/Desktop/smart-visual-inspector/best.pt')
model.to('cpu')  # Ensure the model is on CPU
def detect_defects(img):
    try:
        # Save input image temporarily
        input_path = "temp_input.jpg"
        img.save(input_path)
        
        # Run prediction as in notebook
        results = model.predict(input_path)
        
        # Convert result to image with bounding boxes
        result_img = Image.fromarray(results[0].plot())
        
        return result_img
    except Exception as e:
        print(f"Detection error: {str(e)}")
        raise e