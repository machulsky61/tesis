import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class DebateDataset(Dataset):
    """
    Wrapper dataset that takes a base dataset (complete images) and returns partially revealed examples.
    Each example consists of two channels: [mask, revealed_values] and the original label.
    The mask is binary (1 for revealed pixels, 0 for unrevealed).
    The values plane contains original values only in revealed pixels (0 for the rest).
    Randomly reveals a number of relevant pixels between min_reveal and max_reveal.
    """
    def __init__(self, base_dataset, thr=0.1, min_reveal=1, max_reveal=None):
        self.base = base_dataset
        self.thr = thr
        self.min_reveal = min_reveal
        # If max_reveal is not specified, by default reveals all possible relevant pixels
        self.max_reveal = max_reveal
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        # Get image and label from base dataset
        image, label = self.base[idx]  # image is Tensor 1xHxW (float 0-1), label is integer
        # Calculate mask of relevant pixels according to threshold
        # Consider only intensities > thr as relevant
        # mask_relevant = (image > self.thr)
        # # Extraer coordenadas de píxeles relevantes
        # coords = mask_relevant.nonzero(as_tuple=False)
        # image: Tensor 1×H×W → take channel 0 to have H×W
        original_values = image[0]                         # now H×W
        mask_relevant   = (original_values > self.thr)     # 2D mask
        coords          = mask_relevant.nonzero(as_tuple=False)  # [N, 2]
        # If no relevant pixels (very faint image), consider all pixels as relevant
        if coords.size(0) == 0:
            # generate coords of all pixels
            H, W = image.shape[-2], image.shape[-1]
            coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
        # Determine how many pixels to reveal
        max_reveal = coords.size(0) if self.max_reveal is None else min(self.max_reveal, coords.size(0))
        # Ensure min_reveal doesn't exceed available amount
        min_reveal = min(self.min_reveal, max_reveal)
        num_reveal = random.randint(min_reveal, max_reveal) if max_reveal > 0 else 0
        # Randomly choose num_reveal pixels from coords without replacement
        if num_reveal > 0:
            chosen_indices = random.sample(range(coords.size(0)), num_reveal)
            chosen_coords = coords[chosen_indices]
        else:
            chosen_coords = torch.empty((0,2), dtype=torch.long)
        # Create reveal mask (same dimension as original image, without channel)
        # mask will have 1 for revealed pixels, 0 for the rest
        H, W = image.shape[-2], image.shape[-1]
        reveal_mask = torch.zeros((H, W), dtype=torch.float32)
        for coord in chosen_coords:
            y, x = int(coord[0]), int(coord[1])
            reveal_mask[y, x] = 1.0
        # Create revealed values plane: use original values where mask=1, 0 for the rest
        # image is 1xHxW, extract as 2D for easy multiplication
        original_values = image[0]  # Tensor HxW
        values_plane = original_values * reveal_mask
        # Combine into 2-channel tensor: [mask, values]
        sample_input = torch.stack([reveal_mask, values_plane], dim=0)
        return sample_input, label

def load_datasets(resolution=16, k=6, thr=0.1, batch_size=64):
    """
    Loads MNIST datasets for training and testing, applying given resolution.
    Returns training DataLoader (with partially revealed images) and testing (complete images).
    """
    # Transformations: resize and convert to tensor
    transform_list = []
    if resolution != 28:
        transform_list.append(transforms.Resize((resolution, resolution)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    # Load base MNIST datasets
    train_base = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # Wrap training dataset in DebateDataset to generate partial inputs
    train_dataset = DebateDataset(train_base, thr=thr, min_reveal=k, max_reveal=k)
    # For test, we'll use complete images; don't wrap in DebateDataset since in debate they're revealed interactively
    test_dataset = test_base
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
