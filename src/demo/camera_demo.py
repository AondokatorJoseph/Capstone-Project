import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models.model import WasteClassifier
from src.utils.helpers import load_and_preprocess_image

class WasteClassificationDemo:
    def __init__(self):
        self.classifier = WasteClassifier()
        self.model = self.classifier.build_model()
        
        # Define categories and recyclable materials
        self.categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        self.recyclable = ['glass', 'paper', 'cardboard', 'plastic', 'metal']
    
        # Update weights path
        weights_path = os.path.join(project_root, 'models', 'saved_models', 'waste_classifier.weights.h5')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}. Please train the model first.")
    
        self.model.load_weights(weights_path)
        
    def process_frame(self, frame):
        # Preprocess the frame
        processed = cv2.resize(frame, (224, 224))
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        prediction = self.model.predict(processed)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        material = self.categories[class_idx]
        is_recyclable = material in self.recyclable
        
        return material, confidence, is_recyclable

    def run_demo(self):
        cap = cv2.VideoCapture(0)  # Use default camera
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Make prediction
            material, confidence, is_recyclable = self.process_frame(frame)
            
            # Create display text
            status = "♻️ Recyclable" if is_recyclable else "🗑️ Not Recyclable"
            text = f"Material: {material} ({confidence:.2f})"
            
            # Add text to frame
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if is_recyclable else (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('CyclotekAI Demo', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = WasteClassificationDemo()
    demo.run_demo()