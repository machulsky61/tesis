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

def select_optimal_pixels(image, model, true_label, k, thr, device):
    """
    Selecciona los k píxeles que maximizan la confianza del juez en la clase verdadera.
    Cota superior: mejor caso posible para el juez.
    """
    H, W = image.shape[-2], image.shape[-1]
    image_2d = image.squeeze()
    
    # Obtener píxeles candidatos (>thr para mantener consistencia con entrenamiento)
    mask_relevant = (image_2d > thr)
    coords = mask_relevant.nonzero(as_tuple=False)
    
    if coords.size(0) == 0:
        # Si no hay píxeles relevantes, usar todos
        coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
    
    if coords.size(0) <= k:
        # Si hay menos píxeles candidatos que k, usar todos
        chosen_coords = coords
    else:
        # Evaluar cada píxel individualmente para encontrar los más informativos
        pixel_scores = []
        
        with torch.no_grad():
            for coord in coords:
                y, x = int(coord[0]), int(coord[1])
                # Crear máscara con solo este píxel
                test_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
                test_mask[y, x] = 1.0
                values_plane = image_2d * test_mask
                
                # Evaluar con el juez
                judge_input = torch.stack([test_mask, values_plane], dim=0).unsqueeze(0)
                output = model(judge_input)
                confidence = torch.softmax(output, dim=1)[0, true_label].item()
                pixel_scores.append((confidence, coord))
        
        # Ordenar por confianza descendente y tomar los k mejores
        pixel_scores.sort(key=lambda x: x[0], reverse=True)
        chosen_coords = torch.stack([coord for _, coord in pixel_scores[:k]])
    
    return chosen_coords

def select_adversarial_pixels(image, model, true_label, k, thr, device):
    """
    Selecciona los k píxeles que minimizan la confianza del juez en la clase verdadera.
    Incluye píxeles negros/bajos para testear robustez del juez.
    Cota inferior: peor caso posible para el juez.
    """
    H, W = image.shape[-2], image.shape[-1]
    image_2d = image.squeeze()
    
    # Para adversarial, consideramos TODOS los píxeles, incluyendo los negros
    coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
    
    if coords.size(0) <= k:
        chosen_coords = coords
    else:
        # Evaluar cada píxel para encontrar los que más perjudican al juez
        pixel_scores = []
        
        with torch.no_grad():
            for coord in coords:
                y, x = int(coord[0]), int(coord[1])
                # Crear máscara con solo este píxel
                test_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
                test_mask[y, x] = 1.0
                values_plane = image_2d * test_mask
                
                # Evaluar con el juez
                judge_input = torch.stack([test_mask, values_plane], dim=0).unsqueeze(0)
                output = model(judge_input)
                confidence = torch.softmax(output, dim=1)[0, true_label].item()
                pixel_scores.append((confidence, coord))
        
        # Ordenar por confianza ascendente y tomar los k peores
        pixel_scores.sort(key=lambda x: x[0])
        chosen_coords = torch.stack([coord for _, coord in pixel_scores[:k]])
    
    return chosen_coords

def select_random_pixels(image, k, thr):
    """
    Selecciona k píxeles aleatoriamente de los píxeles relevantes (>thr).
    Estrategia baseline actual.
    """
    H, W = image.shape[-2], image.shape[-1]
    image_2d = image.squeeze()
    
    # Obtener píxeles relevantes
    mask_relevant = (image_2d > thr)
    coords = mask_relevant.nonzero(as_tuple=False)
    
    if coords.size(0) == 0:
        # Si no hay píxeles relevantes, usar todos
        coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
    
    if coords.size(0) <= k:
        chosen_coords = coords
    else:
        # Selección aleatoria
        chosen_indices = random.sample(range(coords.size(0)), k)
        chosen_coords = coords[chosen_indices]
    
    return chosen_coords

def main():
    parser = argparse.ArgumentParser(description="Evalúa la precisión del juez débil (SparseCNN) con diferentes estrategias de selección de píxeles")
    parser.add_argument("--resolution", type=int,   default=28,             help="Resolución de las imágenes (16 o 28)")
    parser.add_argument("--thr",        type=float, default=0.0,            help="Umbral para considerar píxeles relevantes")
    parser.add_argument("--k",          type=int,   default=6,              help="Número exacto de píxeles revelados en cada máscara")
    parser.add_argument("--seed",       type=int,   default=42,             help="Semilla para reproducibilidad")
    parser.add_argument("--n_images",   type=int,   default=1000,           help="Cantidad de muestras de test a evaluar")
    parser.add_argument("--batch_size", type=int,   default=128,            help="Tamaño de batch para evaluación")
    parser.add_argument("--judge_name", type=str,   default="judge_model",  help="Nombre del modelo juez (sin extensión)")
    parser.add_argument("--strategy",   type=str,   default="random",       help="Estrategia de selección de píxeles", 
                        choices=["random", "optimal", "adversarial"])
    parser.add_argument("--note",       type=str,   default="",             help="Nota opcional para registrar en el CSV")
    args = parser.parse_args()

    # Generate descriptive experiment ID for judge evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id = f"eval_{args.judge_name}_{args.strategy}_{args.resolution}px_{args.k}k_{timestamp}"

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

    # 4) Bucle de evaluación con estrategia específica
    total = 0
    correct = 0
    
    strategy_name = {"random": "random", "optimal": "optimal", "adversarial": "adversarial"}[args.strategy]
    
    with torch.no_grad():
        pbar = tqdm(test_subset, desc=f"Evaluating with {strategy_name} strategy")
        for image, label in pbar:
            image = image.to(device)
            true_label = label
            
            # Seleccionar píxeles según la estrategia
            if args.strategy == "random":
                chosen_coords = select_random_pixels(image, args.k, args.thr)
            elif args.strategy == "optimal":
                chosen_coords = select_optimal_pixels(image, model, true_label, args.k, args.thr, device)
            elif args.strategy == "adversarial":
                chosen_coords = select_adversarial_pixels(image, model, true_label, args.k, args.thr, device)
            
            # Crear máscara y entrada para el juez
            H, W = image.shape[-2], image.shape[-1]
            mask = torch.zeros((H, W), dtype=torch.float32, device=device)
            image_2d = image.squeeze()
            
            for coord in chosen_coords:
                y, x = int(coord[0]), int(coord[1])
                mask[y, x] = 1.0
            
            values_plane = image_2d * mask
            judge_input = torch.stack([mask, values_plane], dim=0).unsqueeze(0)
            
            # Evaluar con el juez
            output = model(judge_input)
            pred = output.argmax(dim=1).item()
            
            if pred == true_label:
                correct += 1
            total += 1
            
            pbar.set_postfix({'accuracy': f'{(correct/total)*100:.2f}%'})

    accuracy = correct / total if total > 0 else 0.0
    print(f"Judge accuracy with {strategy_name} strategy and k={args.k} pixels: {accuracy*100:.2f}% "
          f"({correct}/{total})")
    
    # Guardar resultados en evaluations.csv
    os.makedirs("outputs", exist_ok=True)
    results = {
        "timestamp": id,
        "judge_name": args.judge_name,
        "strategy": args.strategy,
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
