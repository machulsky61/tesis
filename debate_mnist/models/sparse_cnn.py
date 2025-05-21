import torch
import torch.nn as nn

class SparseCNN(nn.Module):
    """
    Modelo de juez (clasificador) CNN para MNIST modificado para aceptar dos
    canales de entrada:
    - Canal 0: máscara de píxeles revelados (binaria)
    - Canal 1: valores de los píxeles revelados (0 en los no revelados)
    """
    def __init__(self, resolution=16):
        super(SparseCNN, self).__init__()
        # Capas convolucionales
        # Primer conv: entradas de 2 canales -> 32 filtros
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # reduce a la mitad
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Calcular tamaño de flatten según resolución después de dos pool (dividido por 4)
        self.resolution = resolution
        # asumiendo resolution es divisible por 4 (16 y 28 lo son: 16/4=4, 28/4=7)
        self.flat_dim = 64 * (resolution // 2 // 2) * (resolution // 2 // 2)
        # Capas completamente conectadas
        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, 10) # 10 clases de dígitos
        # Función de activación ReLU (usaremos la misma en forward)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x tiene forma [batch_size, 2, H, W]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Aplanar para FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x