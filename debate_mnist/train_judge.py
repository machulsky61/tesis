import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import data_utils, helpers
from utils.paths import get_model_path, JUDGES_CSV
from models.sparse_cnn import SparseCNN
from tqdm import tqdm
from datetime import datetime
import os

def evaluate_model(
    model_path: str,
    resolution: int = 28,
    thr: float = 0.0,
    k: int = 6,
    seed: int = 42,
    n_samples: int = 10000,
    batch_size: int = 128,
):
    from utils import helpers
    from utils.data_utils import DebateDataset
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, Subset
    import random, torch, numpy as np
    from models.sparse_cnn import SparseCNN

    # 1) Fix seeds
    helpers.set_seed(seed)

    # 2) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCNN(resolution=resolution).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) Dataset y transform
    transform_list = []
    if resolution != 28:
        transform_list.append(transforms.Resize((resolution, resolution)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    full_test = MNIST(root="./data", train=False, download=True, transform=transform)

    # 4) Submuestreo
    indices = list(range(len(full_test)))
    random.shuffle(indices)
    indices = indices[:n_samples]
    test_subset = Subset(full_test, indices)

    # 5) DebateDataset para máscaras de k píxeles
    test_dataset = DebateDataset(
        test_subset,
        thr=thr,
        min_reveal=k,
        max_reveal=k
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 6) Evaluation loop
    total, correct = 0, 0
    with torch.no_grad():
        from tqdm import tqdm
        pbar = tqdm(test_loader, desc="Evaluando juez en train")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'acc': f'{correct/total*100:.2f}%'})

    return correct/total


def main():
    parser = argparse.ArgumentParser(description="Training of judge model (SparseCNN) for MNIST debate")
    parser.add_argument("--resolution", type=int, default=28, help="Input resolution (image size, 16 or 28)")
    parser.add_argument("--k",  type=int,   default=6, help="Fixed number of revealed pixels in training")
    parser.add_argument("--thr", type=float, default=0.0, help="Threshold for relevant pixels")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=64, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--judge_name", type=str, default="judge_model", help="Judge model name (without extension)")
    parser.add_argument("--note", type=str, default="", help="Optional note to record in CSV")
    args = parser.parse_args()
    
    # Generate descriptive experiment ID for judge training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id = f"judge_{args.judge_name}_{args.resolution}px_{args.k}k_{timestamp}"
    
    # Fix seeds for reproducibility
    helpers.set_seed(args.seed)

    # Load MNIST data with specified resolution
    train_loader, test_loader = data_utils.load_datasets(resolution=args.resolution, k=args.k, thr=args.thr, batch_size=args.batch_size)

    # Initialize model and move to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCNN(resolution=args.resolution).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_loss = float('inf')
    training_losses = []
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(train_loader.dataset)
        training_losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save trained model
    model_path = get_model_path(args.judge_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate final model
    final_accuracy = evaluate_model(
        model_path,
        resolution=args.resolution,
        thr=args.thr,
        n_samples=1000,  # Number of test samples to evaluate
        batch_size=args.batch_size
        )
    print(f"Final model accuracy: {final_accuracy*100:.2f}%")
    
    # Save training information to judges.csv
    training_info = {
        "timestamp": id,
        "judge_name": args.judge_name,
        "resolution": args.resolution,
        "thr": args.thr,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_loss": round(best_loss,4),
        "pixels": args.k,
        "accuracy": round(final_accuracy,3),
        "note": args.note
    }
    
    helpers.log_results_csv(str(JUDGES_CSV), training_info)
    print(f"Training information saved to {JUDGES_CSV}")

if __name__ == "__main__":
    main()
