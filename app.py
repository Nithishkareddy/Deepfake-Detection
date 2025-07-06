import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
import tempfile
import os

# Deep Learning Model Definition
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Use pretrained ResNet50 as base model
        self.base_model = resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.base_model(x)

# Video Processing Class
class VideoProcessor:
    def __init__(self, model_path='deepfake_detector_retrained.pth'):  # Updated model path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepfakeDetector().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_frame(self, frame):
        if frame is None:
            return None  # Skip empty frames

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(frame_tensor)
        
        return prediction.item()

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_predictions = []
        frame_count = 0

        if not cap.isOpened():
            return {"error": "Could not open video file"}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame to improve speed
            if frame_count % 5 == 0:
                prediction = self.process_frame(frame)
                if prediction is not None:
                    frame_predictions.append(prediction)
            
            frame_count += 1
        
        cap.release()

        if not frame_predictions:
            return {"error": "No valid frames processed"}

        # Adjust threshold dynamically
        avg_prediction = np.mean(frame_predictions)
        median_prediction = np.median(frame_predictions)
        confidence = abs(0.5 - avg_prediction) * 2  # Scale confidence
        threshold = 0.75  # Adjusted threshold
        is_fake = avg_prediction > threshold

        return {
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'prediction': float(avg_prediction),
            'threshold_used': float(threshold)
        }

# Flask Web Application
app = Flask(__name__)
processor = VideoProcessor()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded video to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_video.mp4')
    video_file.save(temp_path)
    
    try:
        result = processor.analyze_video(temp_path)
        return jsonify(result)
    finally:
        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
