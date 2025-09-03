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
from utils.paths import get_model_path, get_evaluation_folder, EVALUATIONS_CSV

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

def select_adversarial_nonzero_pixels(image, model, true_label, k, thr, device):
    """
    Selecciona los k píxeles que minimizan la confianza del juez en la clase verdadera,
    pero SOLO entre píxeles con valores > thr (sin píxeles negros/zero).
    Cota inferior más realista: peor caso entre píxeles relevantes.
    """
    H, W = image.shape[-2], image.shape[-1]
    image_2d = image.squeeze()
    
    # Solo píxeles relevantes (>thr) - sin píxeles negros
    mask_relevant = (image_2d > thr)
    coords = mask_relevant.nonzero(as_tuple=False)
    
    if coords.size(0) == 0:
        # Si no hay píxeles relevantes, usar todos como fallback
        coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
    
    if coords.size(0) <= k:
        chosen_coords = coords
    else:
        # Evaluar cada píxel relevante para encontrar los que más perjudican al juez
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

def select_random_ood_pixels(image, k, thr):
    """
    Selecciona k píxeles aleatoriamente de TODOS los píxeles (incluyendo negros).
    Estrategia OOD para comparar contra random normal y adversarial OOD.
    Ignora el threshold y considera píxeles negros/background.
    """
    H, W = image.shape[-2], image.shape[-1]
    
    # Para random OOD, consideramos TODOS los píxeles, incluyendo los negros
    coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
    
    if coords.size(0) <= k:
        chosen_coords = coords
    else:
        # Selección aleatoria uniforme de todos los píxeles
        chosen_indices = random.sample(range(coords.size(0)), k)
        chosen_coords = coords[chosen_indices]
    
    return chosen_coords

def select_agent_pixels(image, model, true_label, k, thr, device, agent_type, **agent_kwargs):
    """
    Selecciona k píxeles usando un agente específico (greedy o mcts) de forma secuencial.
    Simula un debate de un solo agente eligiendo k píxeles para el juez.
    
    Args:
        image: imagen original
        model: modelo del juez
        true_label: etiqueta verdadera
        k: número de píxeles a seleccionar
        thr: threshold para píxeles relevantes
        device: dispositivo (GPU/CPU)
        agent_type: "greedy" o "mcts"
        **agent_kwargs: argumentos específicos del agente (rollouts, etc.)
    """
    H, W = image.shape[-2], image.shape[-1]
    image_2d = image.squeeze()
    
    # Inicializar máscara vacía
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Importar agentes dinámicamente
    if agent_type == "greedy":
        from agents.greedy_agent import GreedyAgent
        # Para evaluación del juez: agente honesto que maximiza la clase verdadera
        agent = GreedyAgent(
            judge_model=model,
            my_class=true_label,
            opponent_class=None,  # No hay oponente en evaluación
            precommit=False,      # Modo honesto estándar
            original_image=image,
            thr=thr,
            allow_all_pixels=agent_kwargs.get('allow_all_pixels', False)
        )
    elif agent_type == "greedy_adversarial":
        from agents.greedy_adversarial_agent import GreedyAdversarialAgent
        # Agente adversarial que minimiza logits de la clase verdadera
        agent = GreedyAdversarialAgent(
            judge_model=model,
            true_class=true_label,
            original_image=image,
            thr=thr,
            allow_all_pixels=agent_kwargs.get('allow_all_pixels', False)
        )
    elif agent_type == "mcts":
        from agents.mcts_fast import FastMCTSAgent
        rollouts = agent_kwargs.get('rollouts', 500)
        agent = FastMCTSAgent(
            judge_model=model,
            my_class=true_label,
            opponent_class=None,
            precommit=False,
            original_image=image,
            thr=thr,
            rollouts=rollouts,
            total_moves=k,
            is_truth_agent=True,
            allow_all_pixels=agent_kwargs.get('allow_all_pixels', False)
        )
    elif agent_type == "mcts_adversarial":
        from agents.mcts_adversarial_agent import MCTSAdversarialAgent
        rollouts = agent_kwargs.get('rollouts', 500)
        agent = MCTSAdversarialAgent(
            judge_model=model,
            true_class=true_label,
            original_image=image,
            thr=thr,
            allow_all_pixels=agent_kwargs.get('allow_all_pixels', False),
            rollouts=rollouts
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Selección secuencial de píxeles
    chosen_coords = []
    for turn in range(k):
        pixel = agent.choose_pixel(mask, reveal_count=turn)
        if pixel is None:
            break  # No hay más píxeles disponibles
        
        y, x = pixel
        mask[y, x] = 1.0
        chosen_coords.append(torch.tensor([y, x], device=device))
    
    if chosen_coords:
        return torch.stack(chosen_coords)
    else:
        # Fallback a selección aleatoria si el agente no puede seleccionar
        return select_random_pixels(image, k, thr)

def save_evaluation_outputs(i, image, mask, chosen_coords, logits, true_label, pred, args, eval_id):
    """Saves image, mask and metadata to evaluation output directory."""
    if args.save_images or args.save_masks or args.save_metadata or args.save_visualizations:
        folder_note = args.note.replace(" ", "_") if args.note and len(args.note) < 20 else ""
        eval_folder = get_evaluation_folder(eval_id, folder_note)
    
    if args.save_images or args.save_metadata:
        img_path = os.path.join(eval_folder, f"sample_{i}_image.png")
        helpers.save_image(image, img_path)   
         
    if args.save_masks or args.save_metadata:
        img_path = os.path.join(eval_folder, f"sample_{i}_mask.png")
        helpers.save_mask(image, mask, img_path)
        
    if args.save_metadata:
        # Create comprehensive metadata
        meta = {
            "sample_index": i,
            "true_label": int(true_label),
            "predicted_label": int(pred),
            "is_correct": bool(pred == true_label),
            "strategy": args.strategy,
            "resolution": args.resolution,
            "threshold": args.thr,
            "k_pixels": args.k,
            "pixels_revealed": int(mask.sum().item()),
            "selected_coordinates": [(int(coord[0]), int(coord[1])) for coord in chosen_coords],
            "final_logits": {str(i): float(logits[i]) for i in range(len(logits))},
            "confidence": float(torch.softmax(torch.tensor(logits), dim=0).max().item()),
            "judge_name": args.judge_name,
            "seed": args.seed,
            "rollouts": args.rollouts if hasattr(args, 'rollouts') else None,
            "allow_all_pixels": args.allow_all_pixels,
            "eval_id": eval_id
        }
        
        meta_path = os.path.join(eval_folder, f"sample_{i}_metadata.json")
        helpers.save_play(meta, meta_path)
        
    if args.save_visualizations or args.save_metadata:
        # Create evaluation visualization
        viz_path = os.path.join(eval_folder, f"sample_{i}_evaluation.png")
        
        # Prepare evaluation info for visualization
        eval_info = {
            'eval_id': eval_id,
            'sample_index': i,
            'true_label': int(true_label),
            'predicted_label': int(pred),
            'strategy': args.strategy,
            'accuracy': bool(pred == true_label),
            'confidence': float(torch.softmax(torch.tensor(logits), dim=0).max().item()),
            'logits': {str(i): float(logits[i]) for i in range(len(logits))}
        }
        
        # Convert chosen_coords to list format expected by save_evaluation_visualization
        coords_list = [(int(coord[0]), int(coord[1])) for coord in chosen_coords]
        
        helpers.save_evaluation_visualization(image, mask, coords_list, viz_path, eval_info)

def main():
    parser = argparse.ArgumentParser(description="Evalúa la precisión del juez débil (SparseCNN) con diferentes estrategias de selección de píxeles")
    parser.add_argument("--resolution", type=int,   default=28,             help="Resolución de las imágenes (16 o 28)")
    parser.add_argument("--thr",        type=float, default=0.0,            help="Umbral para considerar píxeles relevantes")
    parser.add_argument("--k",          type=int,   default=6,              help="Número exacto de píxeles revelados en cada máscara")
    parser.add_argument("--seed",       type=int,   default=42,             help="Semilla para reproducibilidad")
    parser.add_argument("--n_images",   type=int,   default=1000,           help="Cantidad de muestras de test a evaluar")
    parser.add_argument("--batch_size", type=int,   default=128,            help="Tamaño de batch para evaluación")
    parser.add_argument("--judge_name", type=str,   default="judge_model",  help="Nombre del modelo juez (sin extensión)")
    parser.add_argument("--strategy",   type=str,   default="random",       help="Pixel selection strategy", 
                        choices=["random", "random_ood", "optimal", "adversarial", "adversarial_nonzero", "greedy_agent", "mcts_agent", "greedy_adversarial_agent", "mcts_adversarial_agent"])
    parser.add_argument("--rollouts",   type=int,   default=500,            help="Rollouts para MCTS agent (solo usado con strategy=mcts_agent)")
    parser.add_argument("--allow_all_pixels", action="store_true",         help="Permitir selección de cualquier píxel (incluye píxeles negros)")
    parser.add_argument("--note",       type=str,   default="",             help="Nota opcional para registrar en el CSV")
    
    # Evaluation output options
    parser.add_argument("--save_images", action="store_true",               help="Save original images for each evaluation sample")
    parser.add_argument("--save_masks", action="store_true",                help="Save masked images for each evaluation sample")
    parser.add_argument("--save_metadata", action="store_true",             help="Save comprehensive metadata (images, masks, and JSON) for each sample")
    parser.add_argument("--save_visualizations", action="store_true",       help="Save colored visualization images showing pixel selection order")
    args = parser.parse_args()

    # Generate descriptive experiment ID for judge evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id = f"eval_{args.judge_name}_{args.strategy}_{args.resolution}px_{args.k}k_{timestamp}"

    # 1) Fijar semillas
    helpers.set_seed(args.seed)

    # 2) Cargar modelo del juez
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCNN(resolution=args.resolution).to(device)
    model_path = get_model_path(args.judge_name)
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
    
    strategy_name = {
        "random": "random", 
        "random_ood": "random_ood",
        "optimal": "optimal", 
        "adversarial": "adversarial", 
        "adversarial_nonzero": "adversarial_nonzero",
        "greedy_agent": "greedy_agent",
        "mcts_agent": "mcts_agent",
        "greedy_adversarial_agent": "greedy_adversarial_agent",
        "mcts_adversarial_agent": "mcts_adversarial_agent"
    }[args.strategy]
    
    with torch.no_grad():
        pbar = tqdm(test_subset, desc=f"Evaluating with {strategy_name} strategy")
        for sample_idx, (image, label) in enumerate(pbar):
            image = image.to(device)
            true_label = label
            
            # Seleccionar píxeles según la estrategia
            if args.strategy == "random":
                chosen_coords = select_random_pixels(image, args.k, args.thr)
            elif args.strategy == "random_ood":
                chosen_coords = select_random_ood_pixels(image, args.k, args.thr)
            elif args.strategy == "optimal":
                chosen_coords = select_optimal_pixels(image, model, true_label, args.k, args.thr, device)
            elif args.strategy == "adversarial":
                chosen_coords = select_adversarial_pixels(image, model, true_label, args.k, args.thr, device)
            elif args.strategy == "adversarial_nonzero":
                chosen_coords = select_adversarial_nonzero_pixels(image, model, true_label, args.k, args.thr, device)
            elif args.strategy == "greedy_agent":
                chosen_coords = select_agent_pixels(
                    image, model, true_label, args.k, args.thr, device, 
                    "greedy", allow_all_pixels=args.allow_all_pixels
                )
            elif args.strategy == "mcts_agent":
                chosen_coords = select_agent_pixels(
                    image, model, true_label, args.k, args.thr, device, 
                    "mcts", rollouts=args.rollouts, allow_all_pixels=args.allow_all_pixels
                )
            elif args.strategy == "greedy_adversarial_agent":
                chosen_coords = select_agent_pixels(
                    image, model, true_label, args.k, args.thr, device, 
                    "greedy_adversarial", allow_all_pixels=args.allow_all_pixels
                )
            elif args.strategy == "mcts_adversarial_agent":
                chosen_coords = select_agent_pixels(
                    image, model, true_label, args.k, args.thr, device, 
                    "mcts_adversarial", rollouts=args.rollouts, allow_all_pixels=args.allow_all_pixels
                )
            
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
            
            # Save outputs if requested
            if args.save_images or args.save_masks or args.save_metadata or args.save_visualizations:
                logits = output[0].cpu().numpy()
                save_evaluation_outputs(
                    sample_idx, image, mask, chosen_coords, logits, 
                    true_label, pred, args, id
                )
            
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
        "rollouts": args.rollouts if args.strategy == "mcts_agent" else "",
        "allow_all_pixels": args.allow_all_pixels,
        "note": args.note
    }
    helpers.log_results_csv(str(EVALUATIONS_CSV), results)
    print(f"Results logged to {EVALUATIONS_CSV}")

if __name__ == "__main__":
    main()
