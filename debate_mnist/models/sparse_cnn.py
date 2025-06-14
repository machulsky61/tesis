import torch
import torch.nn as nn

class SparseCNN(nn.Module):
    """
    Judge model (classifier) CNN for MNIST modified to accept two
    input channels:
    - Channel 0: mask of revealed pixels (binary)
    - Channel 1: values of revealed pixels (0 for unrevealed)
    """
    def __init__(self, resolution=16):
        super(SparseCNN, self).__init__()
        # Convolutional layers
        # First conv: 2-channel inputs -> 32 filters
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # reduce by half
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Calculate flatten size according to resolution after two pools (divided by 4)
        self.resolution = resolution
        # assuming resolution is divisible by 4 (16 and 28 are: 16/4=4, 28/4=7)
        self.flat_dim = 64 * (resolution // 2 // 2) * (resolution // 2 // 2)
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, 10) # 10 digit classes
        # ReLU activation function (we'll use the same in forward)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x has shape [batch_size, 2, H, W]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Flatten for FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x