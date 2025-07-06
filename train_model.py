import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# Define Deepfake Detector Model
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

# Detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
val_data = datasets.ImageFolder(root='dataset/val', transform=transform)

# **Set Batch Size** (Optimized for CPU)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # Batch size 16
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Initialize Model
model = DeepfakeDetector().to(device)

# Loss Function & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 5  # Training for 5 epochs
print("🚀 Training started...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Show batch progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    print(f"✅ Epoch [{epoch+1}/{num_epochs}] Completed! Total Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "deepfake_detector.pth")
print("🎉 Training Complete! Model saved as deepfake_detector.pth")
