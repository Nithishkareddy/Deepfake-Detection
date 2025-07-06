import torch
import torch.nn as nn
from torchvision.models import resnet50

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
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

# Create and save an untrained model
model = DeepfakeDetector()
torch.save(model.state_dict(), 'deepfake_detector.pth')
print("Temporary model file created!")