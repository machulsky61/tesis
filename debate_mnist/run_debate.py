import argparse
import random
import torch
from utils import data_utils, helpers
from models.sparse_cnn import SparseCNN
from agents.greedy_agent import GreedyAgent
# from agents.mcts_agent import MCTSAgent
from agents.mcts_fast import FastMCTSAgent as MCTSAgent
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

def get_agents(agent_type, judge_model, label, opponent_label, precommit, image, thr, rollouts, k):
    """Inicializa los agentes según el tipo."""
    if agent_type == "greedy":
        agent_truth = GreedyAgent(judge_model, label, opponent_label, precommit, image, thr)
        agent_liar = GreedyAgent(judge_model, opponent_label, label, precommit, image, thr)
    else:
        agent_truth = MCTSAgent(judge_model, label, opponent_label, precommit, image, thr, rollouts, k, True)
        agent_liar = MCTSAgent(judge_model, opponent_label, label, precommit, image, thr, rollouts, k, False)
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

        logit_truth = output[0, agent_truth.my_class].item()
    if args.precommit:
        logit_liar  = output[0, agent_liar.my_class].item()
    else:
        logit_liar = "N/A"  # No hay etiqueta fija para el mentiroso en este caso
        
    predicted_label = output[0].argmax().item()  # Elige el label global máximo

    return {
        "mask": mask,
        "predicted_label": predicted_label,
        "logit_truth": logit_truth,
        "logit_liar": logit_liar,
        "logits": output[0].cpu().numpy(),
        "revealed_positions": [(int(y), int(x)) for (y, x) in (mask > 0).nonzero(as_tuple=False).tolist()],
        "pixels_revealed": int(mask.sum().item())
    }
    
def run_exhaustive_precommit(image, label_true, args, judge_model, device):
    """
    Ejecuta el protocolo precommit exhaustivo:
      * 9 etiquetas falsas
      * 3 semillas por cada etiqueta
    Devuelve:
      - winner_label  : etiqueta ganadora tras el procedimiento
      - logits_avg_dct: { etiqueta_falsa: (avg_true, avg_false) }
    """
    logits_avg_dct = {}
    winner_label = label_true  # Por defecto gana el honesto

    wrong_labels = [lbl for lbl in range(10) if lbl != label_true]

    for wrong_lbl in wrong_labels:
        true_accum, false_accum = 0.0, 0.0
        for offset_seed in range(3):
            # Establecer semillas distintas pero reproducibles
            helpers.set_seed(args.seed + offset_seed)

            # Instanciar agentes para ESTA semilla y ESTA etiqueta falsa
            agent_truth, agent_liar = get_agents(
                args.agent_type, judge_model, label_true, wrong_lbl, args.precommit, image, args.thr, args.rollouts, args.k
            )

            res = run_single_debate(
                image, agent_truth, agent_liar, args, device,
            )
            true_accum  += res["logit_truth"]
            false_accum += res["logit_liar"]

        # Promediar
        avg_true  = true_accum  / 3.0
        avg_false = false_accum / 3.0
        logits_avg_dct[wrong_lbl] = (avg_true, avg_false)

        # Si el mentiroso supera al honesto EN ALGUNA etiqueta -> mentiroso gana
        if avg_false > avg_true:
            winner_label = wrong_lbl
            break  # basta con una etiqueta para que pierda la verdad

    return winner_label, logits_avg_dct

def save_outputs(i, image, mask, meta, args, id):
    """Guarda la imagen, máscara y metadatos en el directorio de salida."""
    if args.save_mask or args.save_play or args.save_images or args.save_metadata:
        folder_note = f'_{args.note.replace(" ", "_")}' if args.note and len(args.note) < 20 else ""
        run_folder = f"outputs/debate_{id}{folder_note}"
        os.makedirs(run_folder, exist_ok=True)
    
    if args.save_images or args.save_metadata:
        img_path = os.path.join(run_folder, f"sample_{i}_image.png")
        helpers.save_image(image, img_path)   
         
    if args.save_mask or args.save_metadata:
        img_path = os.path.join(run_folder, f"sample_{i}_mask.png")
        helpers.save_mask(image, mask, img_path)
        
    if args.save_play or args.save_metadata:
        meta_path = os.path.join(run_folder, f"sample_{i}_play.json")
        helpers.save_play(meta, meta_path)

def log_results(args, accuracy, id):
    os.makedirs("models", exist_ok=True)
    results = {
        "timestamp": id,
        "judge_name": args.judge_name,
        "seed": args.seed,
        "resolution": args.resolution,
        "thr": args.thr,
        "rollouts": (args.rollouts if args.agent_type == "mcts" else 0),
        "n_images": args.n_images,
        "agent_type": args.agent_type,
        "pixels": args.k,
        "started": args.starts,
        "precommit": bool(args.precommit),
        "accuracy": accuracy,
        "note": args.note,
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
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--save_play", action="store_true")
    parser.add_argument("--save_metadata", action="store_true", help="Guardar metadatos (imagenes, mascaras y jugadas) de cada debate")
    parser.add_argument("--precommit", action="store_true")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--starts", type=str, choices=["honest", "liar"], default="liar")
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
        image, label_tensor = test_images[i]
        image = image.to(device).squeeze()
        label_true = label_tensor.item()

        if args.precommit:
            winner_label, avg_logits = run_exhaustive_precommit(
                image, label_true, args, judge_model, device
            )
            predicted_label = winner_label
            correct += 1 if predicted_label == label_true else 0
            meta = {
                "index": i,
                "truth_label": int(label_true),
                "winner_label": int(winner_label),
                "avg_logits": {str(lbl): {"true": t, "false": f}
                            for lbl, (t, f) in avg_logits.items()}
            }
            save_outputs(i, image, torch.zeros_like(image[0]), meta, args, id)
        else:
            # No precommit, no hay opponent label fijo.
            agent_truth, agent_liar = get_agents(
                args.agent_type, judge_model, label_true, None, args.precommit,
                image, args.thr, args.rollouts, args.k
            )
            res = run_single_debate(image, agent_truth, agent_liar, args, device)
            predicted_label = res["predicted_label"]
            correct += 1 if predicted_label == label_true else 0

            meta = {
                "index": i,
                "truth_label": int(label_true),
                "logits": {str(i): float(res["logits"][i]) for i in range(10)},
                "pixels_revealed": res["pixels_revealed"],
                "revealed_positions": res["revealed_positions"],
                "predicted_label": int(predicted_label),
                "logit_truth": float(res["logit_truth"]),}
            save_outputs(i, image, res["mask"], meta, args, id)

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Precisión del juez con debate: {accuracy*100:.2f}% (sobre {total} muestras)")
    log_results(args, accuracy, id)

if __name__ == "__main__":
    main()
