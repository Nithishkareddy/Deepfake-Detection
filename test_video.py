import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Load the trained model
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_detector_balanced.pth", map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0

    if not cap.isOpened():
        print("⚠️ Error: Cannot open video file. Check the file path.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(frame_tensor).item()

            print(f"🔍 Frame {frame_count}: Prediction Score = {prediction:.4f}")
            frame_predictions.append(prediction)

        frame_count += 1

    cap.release()

    if not frame_predictions:
        print("⚠️ No frames processed!")
        return

    avg_prediction = np.mean(frame_predictions)
    median_prediction = np.median(frame_predictions)
    confidence = abs(0.5 - avg_prediction) * 2  # Scale confidence
    threshold = avg_prediction - 0.2  # Adjust threshold dynamically
    is_fake = avg_prediction > threshold

    print("\n🔎 **Final Video Analysis**")
    print(f"📊 Average Prediction Score: {avg_prediction:.4f}")
    print(f"📊 Median Prediction Score: {median_prediction:.4f}")
    print(f"💡 Confidence: {confidence * 100:.2f}%")
    print(f"🎯 Adjusted Threshold: {threshold:.4f}")

    if is_fake:
        print("🛑 **Detected as FAKE**")
    else:
        print("✅ **Detected as REAL**")

# Run analysis on a test video
test_video_path = r"C:\Major project\18.mp4"  # Change this to a FAKE video path
analyze_video(test_video_path)
