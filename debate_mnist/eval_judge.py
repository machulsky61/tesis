import argparse
import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from tqdm import tqdm
import os

from models.sparse_cnn import SparseCNN
from utils.data_utils import DebateDataset
from utils import helpers

def main():
    parser = argparse.ArgumentParser(description="Evalúa la precisión del juez débil (SparseCNN) sobre máscaras aleatorias de k píxeles")
    parser.add_argument("--resolution", type=int,   default=28,             help="Resolución de las imágenes (16 o 28)")
    parser.add_argument("--thr",        type=float, default=0.0,            help="Umbral para considerar píxeles relevantes")
    parser.add_argument("--k",          type=int,   default=6,              help="Número exacto de píxeles revelados en cada máscara")
    parser.add_argument("--seed",       type=int,   default=42,             help="Semilla para reproducibilidad")
    parser.add_argument("--n_images",   type=int,   default=10000,          help="Cantidad de muestras de test a evaluar")
    parser.add_argument("--batch_size", type=int,   default=128,            help="Tamaño de batch para evaluación")
    parser.add_argument("--judge_name", type=str,   default="judge_model",  help="Nombre del modelo juez (sin extensión)")
    parser.add_argument("--note",       type=str,   default="",             help="Nota opcional para registrar en el CSV")
    args = parser.parse_args()

    #id numeric based on timestamp
    id = int(datetime.now().strftime("%Y%m%d-%H%M%S").replace("-", "").replace(":", ""))

    # 1) Fijar semillas
    helpers.set_seed(args.seed)

    # 2) Cargar modelo del juez
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCNN(resolution=args.resolution).to(device)
    model_path = f"models/{args.judge_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) Preparar transform y dataset base de MNIST
    transform_list = []
    if args.resolution != 28:
        transform_list.append(transforms.Resize((args.resolution, args.resolution)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    full_test = MNIST(root="./data", train=False, download=True, transform=transform)
    # Submuestrear para n_images
    indices = list(range(len(full_test)))
    random.shuffle(indices)
    indices = indices[: args.n_images]
    test_subset = Subset(full_test, indices)

    # 4) Envolver en DebateDataset para generar máscaras de exactamente k píxeles
    test_dataset = DebateDataset(
        test_subset,
        thr=args.thr,
        min_reveal=args.k,
        max_reveal=args.k
    )

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=2)

    # 5) Bucle de evaluación
    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluando")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            pbar.set_postfix({'accuracy': f'{(correct/total)*100:.2f}%'})

    accuracy = correct / total if total > 0 else 0.0
    print(f"Baseline accuracy del juez débil con k={args.k} píxeles: {accuracy*100:.2f}% "
          f"({correct}/{total})")
    
    # Guardar resultados en evaluations.csv
    os.makedirs("models", exist_ok=True)
    results = {
        "timestamp": id,
        "judge_name": args.judge_name,
        "resolution": args.resolution,
        "thr": args.thr,
        "seed": args.seed,
        "n_images": args.n_images,
        "pixels": args.k,
        "accuracy": accuracy,
        "note": args.note
    }
    evaluations_csv_path = "outputs/evaluations.csv"
    helpers.log_results_csv(evaluations_csv_path, results)
    print(f"Resultados registrados en {evaluations_csv_path}")

if __name__ == "__main__":
    main()
