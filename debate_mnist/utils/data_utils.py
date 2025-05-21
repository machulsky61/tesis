import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class DebateDataset(Dataset):
    """
    Dataset envoltorio que toma un dataset base (imágenes completas) y devuelve ejemplos parcialmente revelados.
    Cada ejemplo consiste en dos canales: [máscara, valores_revelados] y la etiqueta original.
    La máscara es binaria (1 en píxeles revelados, 0 en no revelados).
    El plano de valores contiene los valores originales solo en píxeles revelados (0 en el resto).
    Se revela aleatoriamente una cantidad de píxeles relevantes entre min_reveal y max_reveal.
    """
    def __init__(self, base_dataset, thr=0.1, min_reveal=1, max_reveal=None):
        self.base = base_dataset
        self.thr = thr
        self.min_reveal = min_reveal
        # Si max_reveal no se especifica, por defecto revela todos los píxeles relevantes posibles
        self.max_reveal = max_reveal
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        # Obtener imagen y etiqueta del dataset base
        image, label = self.base[idx]  # image es Tensor 1xHxW (float 0-1), label es entero
        # Calcular máscara de píxeles relevantes según el umbral
        # Considerar solo intensidades > thr como relevantes
        # mask_relevant = (image > self.thr)
        # # Extraer coordenadas de píxeles relevantes
        # coords = mask_relevant.nonzero(as_tuple=False)
        # image: Tensor 1×H×W → tomamos el canal 0 para tener H×W
        original_values = image[0]                         # ahora H×W
        mask_relevant   = (original_values > self.thr)     # máscara 2D
        coords          = mask_relevant.nonzero(as_tuple=False)  # [N, 2]
        # Si no hay píxeles relevantes (imagen muy tenue), considerar todos los píxeles como relevantes
        if coords.size(0) == 0:
            # generar coords de todos los píxeles
            H, W = image.shape[-2], image.shape[-1]
            coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
        # Determinar cuántos píxeles revelar
        max_reveal = coords.size(0) if self.max_reveal is None else min(self.max_reveal, coords.size(0))
        # Asegurarse de que min_reveal no exceda la cantidad disponible
        min_reveal = min(self.min_reveal, max_reveal)
        num_reveal = random.randint(min_reveal, max_reveal) if max_reveal > 0 else 0
        # Elegir aleatoriamente num_reveal píxeles de coords sin reemplazo
        if num_reveal > 0:
            chosen_indices = random.sample(range(coords.size(0)), num_reveal)
            chosen_coords = coords[chosen_indices]
        else:
            chosen_coords = torch.empty((0,2), dtype=torch.long)
        # Crear máscara de revelado (misma dimensión que imagen original, sin canal)
        # mask tendrá 1 en píxeles revelados, 0 en el resto
        H, W = image.shape[-2], image.shape[-1]
        reveal_mask = torch.zeros((H, W), dtype=torch.float32)
        for coord in chosen_coords:
            y, x = int(coord[0]), int(coord[1])
            reveal_mask[y, x] = 1.0
        # Crear plano de valores revelados: usar valores originales donde mask=1, 0 en el resto
        # image es 1xHxW, extraerla como 2D para multiplicar fácilmente
        original_values = image[0]  # Tensor HxW
        values_plane = original_values * reveal_mask
        # Combinar en tensor de 2 canales: [mask, values]
        sample_input = torch.stack([reveal_mask, values_plane], dim=0)
        return sample_input, label

def load_datasets(resolution=16, k=6, thr=0.1, batch_size=64):
    """
    Carga los datasets de MNIST para entrenamiento y prueba, aplicando la resolución dada.
    Devuelve DataLoader de entrenamiento (con imágenes parcialmente reveladas) y de prueba (imágenes completas).
    """
    # Transformaciones: redimensionar y convertir a tensor
    transform_list = []
    if resolution != 28:
        transform_list.append(transforms.Resize((resolution, resolution)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    # Cargar datasets base de MNIST
    train_base = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # Envolver el dataset de entrenamiento en DebateDataset para generar entradas parciales
    train_dataset = DebateDataset(train_base, thr=thr, min_reveal=k, max_reveal=k)
    # Para test, usaremos las imágenes completas; no envolvemos en DebateDataset ya que en el debate se revelan interactivamente
    test_dataset = test_base
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
