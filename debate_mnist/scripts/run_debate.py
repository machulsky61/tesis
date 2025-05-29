import argparse
import random
import torch
from utils import data_utils, helpers
from models.sparse_cnn import SparseCNN
from agents.greedy_agent import GreedyAgent
from agents.mcts_agent import MCTSAgent
from datetime import datetime
from tqdm import tqdm
import os

def load_judge_model(judge_name, resolution, device):
    """Carga el modelo juez desde disco."""
    judge_model = SparseCNN(resolution=resolution).to(device)
    model_path = f"models/{judge_name}.pth"
    judge_model.load_state_dict(torch.load(model_path, map_location=device))
    judge_model.eval()
    return judge_model

def get_agents(agent_type, judge_model, label, opponent_label, image, thr, rollouts, k):
    """Inicializa los agentes según el tipo."""
    if agent_type == "greedy":
        agent_truth = GreedyAgent(judge_model, label, opponent_label, image, thr=thr)
        agent_liar = GreedyAgent(judge_model, opponent_label, label, image, thr=thr)
    else:
        agent_truth = MCTSAgent(judge_model, label, opponent_label, image, thr=thr,
                                rollouts=rollouts, total_moves=k, is_truth_agent=True)
        agent_liar = MCTSAgent(judge_model, opponent_label, label, image, thr=thr,
                               rollouts=rollouts, total_moves=k, is_truth_agent=False)
    return agent_truth, agent_liar

def run_single_debate(image, agent_truth, agent_liar, args, device):
    """
    Ejecuta un único debate:
    - Alterna turnos según el agente inicial (args.starts).
    - Cada agente juega su movimiento hasta completar k movimientos o agotar opciones.
    - Devuelve mask final, logits, predicted_label, etc.
    """
    H, W = image.shape[-2], image.shape[-1]
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Definir orden de agentes
    if args.starts == "honest":
        agent_order = [agent_truth, agent_liar]
    else:
        agent_order = [agent_liar, agent_truth]
    
    # Alternar turnos
    for move in range(args.k):
        current_agent_idx = move % 2
        current_agent = agent_order[current_agent_idx]
        pixel = current_agent.choose_pixel(mask, reveal_count=move)
        if pixel is None:
            break
        y, x = pixel
        mask[y, x] = 1.0

    # Calcular output del juez
    values_plane = image * mask
    judge_input = torch.stack([mask, values_plane], dim=0).unsqueeze(0)
    with torch.no_grad():
        output = agent_truth.judge(judge_input)  # Ambos agentes comparten el mismo modelo    
    # --- PRECOMMIT: solo dos clases ---
    if args.precommit:
        logit_truth = output[0, agent_truth.my_class].item()
        logit_liar  = output[0, agent_liar.my_class].item()
        if logit_truth >= logit_liar:
            predicted_label = agent_truth.my_class
        else:
            predicted_label = agent_liar.my_class
    else:
    # --- NO PRECOMMIT: máximo entre todas las clases ---
        logits = output[0]
        predicted_label = logits.argmax().item()  # Elige el label global máximo
        logit_truth = output[0, agent_truth.my_class].item()
        # logit_liar = output[0, agent_liar.my_class].item()

    return {
        "mask": mask,
        "predicted_label": predicted_label,
        "logit_truth": logit_truth,
        "logit_liar": logit_liar if args.precommit else None,
        "revealed_positions": [(int(y), int(x)) for (y, x) in (mask > 0).nonzero(as_tuple=False).tolist()],
        "pixels_revealed": int(mask.sum().item())
    }

def save_outputs(i, image, mask, meta, args, id):
    """Guarda la imagen, máscara y metadatos en el directorio de salida."""
    folder_note = f'_{args.note.replace(" ", "_")}' if args.note and len(args.note) < 20 else ""
    run_folder = f"outputs/debate_{id}{folder_note}"
    os.makedirs(run_folder, exist_ok=True)
    if args.save_images:
            img_path = os.path.join(run_folder, f"sample_{i}.png")
            helpers.save_image(image, mask, img_path)
    if args.save_metadata:
            meta_path = os.path.join(run_folder, f"sample_{i}.json")
            helpers.save_metadata(meta, meta_path)

def log_results(args, accuracy, id):
    os.makedirs("models", exist_ok=True)
    results = {
        "timestamp": id,
        "judge_name": args.judge_name,
        "seed": args.seed,
        "resolution": args.resolution,
        "thr": args.thr,
        "k": args.k,
        "agent_type": args.agent_type,
        "rollouts": (args.rollouts if args.agent_type == "mcts" else 0),
        "n_images": args.n_images,
        "accuracy": accuracy,
        "precommit": bool(args.precommit),
        "note": args.note,
        "started": args.starts
    }
    debates_csv_path = "outputs/debates.csv"
    helpers.log_results_csv(debates_csv_path, results)
    print(f"Resultados registrados en {debates_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimento de debate AI Safety via Debate en MNIST")
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--thr", type=float, default=0.0)
    parser.add_argument("--agent_type", type=str, choices=["greedy", "mcts"], default="greedy")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge_name", type=str, default="judge_model")
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_metadata", action="store_true")
    parser.add_argument("--precommit", action="store_true")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--starts", type=str, choices=["honest", "liar"], default="honest")
    args = parser.parse_args()

    #id numeric based on timestamp
    id = int(datetime.now().strftime("%Y%m%d-%H%M%S").replace("-", "").replace(":", ""))
    
    helpers.set_seed(args.seed)
    _, test_loader = data_utils.load_datasets(resolution=args.resolution, thr=args.thr, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge_model = load_judge_model(args.judge_name, args.resolution, device)

    total, correct = 0, 0

    # Convert test_loader en una lista (para poder samplear)
    test_images = list(test_loader)
    if len(test_images) < args.n_images:
        raise ValueError(f"Solo hay {len(test_images)} imágenes, pero pediste {args.n_images}")

    for i in tqdm(range(args.n_images), desc="Debates"):
        image, label = test_images[i]
        image = image.to(device).squeeze()
        label = label.item()
        classes = list(range(10))
        classes.remove(label)
        opponent_label = random.choice(classes)
        agent_truth, agent_liar = get_agents(
            args.agent_type, judge_model, label, opponent_label, image, args.thr, args.rollouts, args.k
        )
        results = run_single_debate(image, agent_truth, agent_liar, args, device)
        predicted_label = results["predicted_label"]
        if predicted_label == label:
            correct += 1
        total += 1

        meta = {
            "index": i,
            "truth_label": int(label),
            "liar_label": int(opponent_label) if args.precommit else None,
            "pixels_revealed": results["pixels_revealed"],
            "revealed_positions": results["revealed_positions"],
            "predicted_label": int(predicted_label),
            "logits": {str(label): results["logit_truth"], str(opponent_label): results["logit_liar"]}
        }
        save_outputs(i, image, results["mask"], meta, args, id)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Precisión del juez con debate: {accuracy*100:.2f}% (sobre {total} muestras)")
    log_results(args, accuracy, id)

if __name__ == "__main__":
    main()
