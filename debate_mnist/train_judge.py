import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import data_utils, helpers
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

    # 1) Fijar semillas
    helpers.set_seed(seed)

    # 2) Cargar modelo
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

    # 6) Bucle de evaluación
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
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo juez (SparseCNN) para debate en MNIST")
    parser.add_argument("--resolution", type=int, default=28, help="Resolución de entrada (tamaño de imagen, 16 o 28)")
    parser.add_argument("--k",  type=int,   default=6, help="Número fijo de píxeles revelados en entrenamiento")
    parser.add_argument("--thr", type=float, default=0.0, help="Umbral para píxeles relevantes")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--epochs", type=int, default=64, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamaño de batch para entrenamiento")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate para el optimizador")
    parser.add_argument("--judge_name", type=str, default="judge_model", help="Nombre del modelo juez (sin extensión)")
    parser.add_argument("--note", type=str, default="", help="Nota opcional para registrar en el CSV")
    args = parser.parse_args()
    
    # Generate descriptive experiment ID for judge training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id = f"judge_{args.judge_name}_{args.resolution}px_{args.k}k_{timestamp}"
    
    # Fijar semillas para reproducibilidad
    helpers.set_seed(args.seed)

    # Cargar datos de MNIST con la resolución especificada
    train_loader, test_loader = data_utils.load_datasets(resolution=args.resolution, k=args.k, thr=args.thr, batch_size=args.batch_size)

    # Inicializar modelo y mover a dispositivo (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCNN(resolution=args.resolution).to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Bucle de entrenamiento
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

    # Guardar modelo entrenado
    model_path = f"models/{args.judge_name}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

    # Evaluar el modelo final
    final_accuracy = evaluate_model(
        model_path,
        resolution=args.resolution,
        thr=args.thr,
        n_samples=1000,  # Cantidad de muestras de test a evaluar
        batch_size=args.batch_size
        )
    print(f"Precisión final del modelo: {final_accuracy*100:.2f}%")
    
    # Guardar información del entrenamiento en judges.csv
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
    
    judges_csv_path = "outputs/judges.csv"
    helpers.log_results_csv(judges_csv_path, training_info)
    print(f"Información de entrenamiento guardada en {judges_csv_path}")

if __name__ == "__main__":
    main()
