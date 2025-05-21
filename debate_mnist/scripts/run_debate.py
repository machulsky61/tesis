import argparse
import random
import torch
from utils import data_utils, helpers
from models.sparse_cnn import SparseCNN
from agents.greedy_agent import GreedyAgent
# from agents.mcts_agent import MCTSAgent
from agents.mtcs_fast import FastMCTSAgent as MCTSAgent
from datetime import datetime
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimento de debate AI Safety via Debate en MNIST")
    parser.add_argument("--resolution",         type=int,                               default=28,                         help="Resolución de las imágenes (debe coincidir con la resolución usada al entrenar el juez)")
    parser.add_argument("--thr",                type=float,                             default=0.0,                        help="Umbral para considerar píxeles relevantes")
    parser.add_argument("--agent_type",         type=str,   choices=["greedy", "mcts"], default="greedy",                   help="Tipo de agente para ambos debatientes (greedy o mcts)")
    parser.add_argument("--rollouts",           type=int,                               default=50,                         help="Número de simulaciones MCTS (solo aplica si agent_type = 'mcts')")
    parser.add_argument("--k",                  type=int,                               default=6,                          help="Número total de revelaciones permitidas (ej. 4 implica 2 por agente si se alternan equitativamente)")
    parser.add_argument("--seed",               type=int,                               default=42,                         help="Semilla para reproducibilidad del experimento")
    parser.add_argument("--judge_name",         type=str,                               default="judge_model",              help="Nombre del modelo juez (sin extensión)")
    parser.add_argument("--n_images",           type=int,                               default=100,                        help="Número de muestras de test a debatir (por defecto 100)")
    parser.add_argument("--save_images",        action="store_true",                                                        help="Si se especifica, guarda imágenes PNG del estado final de cada debate")
    parser.add_argument("--save_metadata",      action="store_true",                                                        help="Si se especifica, guarda metadatos JSON por cada debate")
    parser.add_argument("--precommit",          action="store_true",                                                        help="Si se especifica, se usara precommit como la manera en que se decide el ganador del debate")
    parser.add_argument("--note",               type=str,                               default="",                         help="Nota opcional para registrar en el CSV")
    parser.add_argument("--starts",             type=str,   choices=["honest", "liar"], default="honest",                   help="Quién mueve primero en el debate (honest o liar)")
    args = parser.parse_args()
    # Fijar semillas para reproducibilidad
    helpers.set_seed(args.seed)
    # Cargar dataset de prueba (imágenes completas) con resolución adecuada
    _, test_loader = data_utils.load_datasets(resolution=args.resolution, thr=args.thr, batch_size=1)
    # Cargar modelo del juez
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge_model = SparseCNN(resolution=args.resolution).to(device)
    model_path = f"models/{args.judge_name}.pth"
    judge_model.load_state_dict(torch.load(model_path, map_location=device))
    judge_model.eval()
    # Variables para conteo de resultados
    total = 0
    correct = 0
    # Iterar sobre muestras de prueba
    for i, (image, label) in enumerate(tqdm(test_loader, desc="Debates")):
        if i >= args.n_images:
            break
        image = image.to(device)
        image = image.squeeze()
        label = label.item()  # clase verdadera
        # Seleccionar aleatoriamente la clase oponente distinta de la verdadera
        classes = list(range(10))
        classes.remove(label)
        opponent_label = random.choice(classes)
        # Inicializar agentes
        # Agente de la verdad (clase real) y agente engañoso (clase oponente)
        if args.agent_type == "greedy":
            agent_truth = GreedyAgent(judge_model, label, opponent_label, image, thr=args.thr)
            agent_liar = GreedyAgent(judge_model, opponent_label, label, image, thr=args.thr)
        else:  # "mcts"
            # Proporcionar total de movimientos k y rol (is_truth_agent) adecuado
            agent_truth = MCTSAgent(judge_model, label, opponent_label, image, thr=args.thr, simulations=args.rollouts, total_moves=args.k, is_truth_agent=True)
            agent_liar = MCTSAgent(judge_model, opponent_label, label, image, thr=args.thr, simulations=args.rollouts, total_moves=args.k, is_truth_agent=False)
        # Estado inicial: ninguna revelación
        H, W = image.shape[-2], image.shape[-1]
        mask = torch.zeros((H, W), dtype=torch.float32, device=device)
        # Determinar cuántos movimientos hará cada agente (distribución de k)
        moves_agent_truth = args.k // 2 if args.k % 2 == 0 else (args.k // 2 + 1)
        moves_agent_liar = args.k // 2
        # Determinar quién mueve primero según el flag
        first_mover_is_truth = args.starts == "honest"
        # Asumimos que el agente de la verdad inicia primero
        reveal_count = 0
        for move in range(args.k):
            # Determinar si es turno del agente de la verdad basado en first_move
            is_truth_turn = (move % 2 == 0) if first_mover_is_truth else (move % 2 == 1)
            if is_truth_turn:
                # Turno del agente de la verdad
                pixel = agent_truth.choose_pixel(mask, reveal_count=reveal_count)
            else:
                # Turno del agente engañoso
                pixel = agent_liar.choose_pixel(mask, reveal_count=reveal_count)
            if pixel is None:
                # No hay más movimientos posibles (todos pix relevantes revelados)
                break
            # Revelar el pixel en la máscara global
            y, x = pixel
            mask[y, x] = 1.0
            reveal_count += 1
        # Una vez terminadas las revelaciones, consultar al juez
        values_plane = image * mask  # aplicar máscara sobre imagen original
        judge_input = torch.stack([mask, values_plane], dim=0).unsqueeze(0)  # input shape [1, 2, H, W]
        with torch.no_grad():
            output = judge_model(judge_input)
        # Obtener logits para las dos clases en disputa
        logit_truth = output[0, label].item()
        logit_liar = output[0, opponent_label].item()
        # Decidir clase elegida por el juez
        if args.precommit:
            predicted_label = label if logit_truth >= logit_liar else opponent_label
        else:
            predicted_label = output.argmax(dim=1).item()
        # Verificar si el juez acertó (eligió la clase verdadera)
        if predicted_label == label:
            correct += 1
        total += 1
        # Guardar imagen y metadatos si se solicita
        if args.save_images:
            os.makedirs("outputs/images", exist_ok=True)
            img_path = f"outputs/images/sample{i}_res{args.resolution}_k{args.k}_{args.agent_type}.png"
            helpers.save_image(image, mask, img_path)
        if args.save_metadata:
            os.makedirs("outputs/metadata", exist_ok=True)
            meta = {
                "index": i,
                "truth_label": int(label),
                "liar_label": int(opponent_label),
                "pixels_revealed": int(mask.sum().item()),
                "revealed_positions": [(int(y), int(x)) for (y, x) in (mask > 0).nonzero(as_tuple=False).tolist()],
                "predicted_label": int(predicted_label),
                "logits": {str(label): logit_truth, str(opponent_label): logit_liar}
            }
            meta_path = f"outputs/metadata/sample{i}_res{args.resolution}_k{args.k}_{args.agent_type}.json"
            helpers.save_metadata(meta, meta_path)
    # Calcular precisión del juez con debate
    accuracy = correct / total if total > 0 else 0.0
    print(f"Precisión del juez con debate: {accuracy*100:.2f}% (sobre {total} muestras)")
    # Registrar resultados en debates.csv
    os.makedirs("models", exist_ok=True)
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_name": args.judge_name,
        "seed": args.seed,
        "resolution": args.resolution,
        "thr": args.thr,
        "k": args.k,
        "agent_type": args.agent_type,
        "rollouts": (args.rollouts if args.agent_type == "mcts" else 0),
        "n_images": args.n_images,
        "accuracy": accuracy,
        "precommit": True if args.precommit else False,
        "note": args.note,
        "started": args.starts
    }
    debates_csv_path = "models/debates.csv"
    helpers.log_results_csv(debates_csv_path, results)
    print(f"Resultados registrados en {debates_csv_path}")

if __name__ == "__main__":
    main()
