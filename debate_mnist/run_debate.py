import argparse
import random
import torch
import numpy as np
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

def get_agents(agent_type, judge_model, label, opponent_label, precommit, image, thr, rollouts, k, mixed_agents=False, honest_agent=None, allow_all_pixels=False):
    """Inicializa los agentes según el tipo."""
    if mixed_agents:
        # Modo mixto: un agente MCTS y otro Greedy
        if honest_agent == "mcts":
            agent_truth = MCTSAgent(judge_model, label, opponent_label, precommit, image, thr, rollouts, k, True, allow_all_pixels)
            agent_liar = GreedyAgent(judge_model, opponent_label, label, precommit, image, thr, allow_all_pixels)
        else:  # honest_agent == "greedy"
            agent_truth = GreedyAgent(judge_model, label, opponent_label, precommit, image, thr, allow_all_pixels)
            agent_liar = MCTSAgent(judge_model, opponent_label, label, precommit, image, thr, rollouts, k, False, allow_all_pixels)
    else:
        # Modo normal: ambos agentes del mismo tipo
        if agent_type == "greedy":
            agent_truth = GreedyAgent(judge_model, label, opponent_label, precommit, image, thr, allow_all_pixels)
            agent_liar = GreedyAgent(judge_model, opponent_label, label, precommit, image, thr, allow_all_pixels)
        else:
            agent_truth = MCTSAgent(judge_model, label, opponent_label, precommit, image, thr, rollouts, k, True, allow_all_pixels)
            agent_liar = MCTSAgent(judge_model, opponent_label, label, precommit, image, thr, rollouts, k, False, allow_all_pixels)
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
        agent_names = ["honest", "liar"]
    else:
        agent_order = [agent_liar, agent_truth]
        agent_names = ["liar", "honest"]
    
    # Rastrear movimientos para visualización coloreada
    debate_moves = []
    
    # Rastrear confianza del juez si está habilitado
    confidence_progression = []
    if args.track_confidence:
        # Evaluar estado inicial (sin píxeles revelados)
        values_plane_init = torch.zeros_like(mask)
        judge_input_init = torch.stack([mask, values_plane_init], dim=0).unsqueeze(0)
        with torch.no_grad():
            output_init = agent_truth.judge(judge_input_init)
            logits_init = output_init[0].cpu().numpy()
            probs_init = torch.softmax(output_init[0], dim=0).cpu().numpy()
            
            confidence_progression.append({
                "turn": 0,
                "pixels_revealed": 0,
                "agent": "initial",
                "pixel_revealed": None,
                "logits": {str(i): float(logits_init[i]) for i in range(10)},
                "probabilities": {str(i): float(probs_init[i]) for i in range(10)},
                "predicted_class": int(output_init[0].argmax().item()),
                "max_confidence": float(probs_init.max())
            })
    
    # Alternar turnos
    for move in range(args.k):
        current_agent_idx = move % 2
        current_agent = agent_order[current_agent_idx]
        current_agent_name = agent_names[current_agent_idx]
        
        pixel = current_agent.choose_pixel(mask, reveal_count=move)
        if pixel is None:
            break
        y, x = pixel
        mask[y, x] = 1.0
        
        # Guardar movimiento para visualización
        debate_moves.append((y, x, current_agent_name, move + 1))
        
        # Trackear confianza del juez después de este movimiento
        if args.track_confidence:
            values_plane_current = image * mask
            judge_input_current = torch.stack([mask, values_plane_current], dim=0).unsqueeze(0)
            with torch.no_grad():
                output_current = agent_truth.judge(judge_input_current)
                logits_current = output_current[0].cpu().numpy()
                probs_current = torch.softmax(output_current[0], dim=0).cpu().numpy()
                
                confidence_progression.append({
                    "turn": move + 1,
                    "pixels_revealed": int(mask.sum().item()),
                    "agent": current_agent_name,
                    "pixel_revealed": (int(y), int(x)),
                    "logits": {str(i): float(logits_current[i]) for i in range(10)},
                    "probabilities": {str(i): float(probs_current[i]) for i in range(10)},
                    "predicted_class": int(output_current[0].argmax().item()),
                    "max_confidence": float(probs_current.max())
                })

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

    # Para modo no-precommit, calcular segundo logit más alto
    second_highest_logit = None
    second_highest_class = None
    if not args.precommit:
        logits_sorted = output[0].cpu().numpy()
        sorted_indices = np.argsort(logits_sorted)[::-1]  # Orden descendente
        if len(sorted_indices) >= 2:
            second_highest_class = int(sorted_indices[1])  # Convertir a int
            second_highest_logit = float(logits_sorted[second_highest_class])  # Convertir a float

    result = {
        "mask": mask,
        "predicted_label": predicted_label,
        "logit_truth": logit_truth,
        "logit_liar": logit_liar,
        "logits": output[0].cpu().numpy(),
        "revealed_positions": [(int(y), int(x)) for (y, x) in (mask > 0).nonzero(as_tuple=False).tolist()],
        "pixels_revealed": int(mask.sum().item()),
        "debate_moves": debate_moves,
        "liar_class": agent_liar.my_class if hasattr(agent_liar, 'my_class') else None,
        "second_highest_logit": second_highest_logit,
        "second_highest_class": second_highest_class
    }
    
    # Agregar progresión de confianza si está habilitada
    if args.track_confidence:
        result["confidence_progression"] = confidence_progression
    
    return result
    
def run_exhaustive_precommit(image, label_true, args, judge_model, device):
    """
    Ejecuta el protocolo precommit exhaustivo:
      * 9 etiquetas falsas
      * 3 semillas por cada etiqueta
    Devuelve:
      - winner_label  : etiqueta ganadora tras el procedimiento
      - logits_avg_dct: { etiqueta_falsa: (avg_true, avg_false) }
      - winning_logits: (logit_truth, logit_liar) para los valores que determinaron la victoria
    """
    logits_avg_dct = {}
    winner_label = label_true  # Por defecto gana el honesto
    winning_logits = (0.0, 0.0)  # Default para cuando gana el honesto

    wrong_labels = [lbl for lbl in range(10) if lbl != label_true]

    for wrong_lbl in wrong_labels:
        true_accum, false_accum = 0.0, 0.0
        for offset_seed in range(3):
            # Establecer semillas distintas pero reproducibles
            helpers.set_seed(args.seed + offset_seed)

            # Instanciar agentes para ESTA semilla y ESTA etiqueta falsa
            agent_truth, agent_liar = get_agents(
                args.agent_type, judge_model, label_true, wrong_lbl, args.precommit, image, args.thr, args.rollouts, args.k,
                args.mixed_agents, args.honest_agent, args.allow_all_pixels
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
            winning_logits = (avg_true, avg_false)  # Guardar los logits que determinaron la victoria
            break  # basta con una etiqueta para que pierda la verdad

    # Si ningún mentiroso ganó, usar los logits del honesto vs el mejor mentiroso
    if winner_label == label_true and logits_avg_dct:
        # Encontrar el mejor desafío del mentiroso
        best_liar_class = max(logits_avg_dct.keys(), key=lambda k: logits_avg_dct[k][1])
        winning_logits = logits_avg_dct[best_liar_class]

    return winner_label, logits_avg_dct, winning_logits

def save_outputs(i, image, mask, meta, args, id):
    """Guarda la imagen, máscara y metadatos en el directorio de salida."""
    if args.save_mask or args.save_play or args.save_images or args.save_metadata or args.save_colored_debate or args.track_confidence:
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
        
    if args.save_colored_debate or args.save_metadata:
        if 'debate_moves' in meta:
            colored_path = os.path.join(run_folder, f"sample_{i}_colored_debate.png")
            
            # Preparar información del debate
            predicted_label = meta.get('predicted_label', 0)
            
            # Calcular predicted_logit correctamente
            if 'logits' in meta and isinstance(meta['logits'], dict):
                predicted_logit = meta['logits'].get(str(predicted_label), 0.0)
            elif meta.get('predicted_label') == meta.get('truth_label'):
                # Si es correcto, usar honest_logit
                predicted_logit = meta.get('logit_truth', 0.0)
            elif 'liar_logit' in meta and meta['liar_logit'] != "N/A":
                # Si es incorrecto y tenemos liar_logit, usarlo
                predicted_logit = meta.get('liar_logit', 0.0)
            else:
                predicted_logit = 0.0
            
            debate_info = {
                'run_id': str(id),
                'sample_index': i,
                'true_label': meta.get('truth_label', '?'),
                'predicted_label': predicted_label,
                'predicted_logit': predicted_logit,
                'honest_logit': meta.get('logit_truth', 0.0),
            }
            
            # Determinar tipos de agentes
            if args.mixed_agents:
                debate_info['honest_agent_type'] = args.honest_agent
                debate_info['liar_agent_type'] = 'mcts' if args.honest_agent == 'greedy' else 'greedy'
            else:
                debate_info['honest_agent_type'] = args.agent_type
                debate_info['liar_agent_type'] = args.agent_type
            
            # Información adicional del debate
            debate_info['first_move'] = args.starts
            
            # Agregar información del mentiroso si está disponible
            if 'liar_class' in meta and meta['liar_class'] is not None:
                debate_info['liar_label'] = meta['liar_class']
                if 'logit_liar' in meta and meta['logit_liar'] != "N/A":
                    debate_info['liar_logit'] = meta['logit_liar']
            
            # Agregar segundo logit más alto para modo no-precommit
            if not args.precommit:
                if 'second_highest_logit' in meta and meta['second_highest_logit'] is not None:
                    debate_info['second_highest_logit'] = meta['second_highest_logit']
                    debate_info['second_highest_class'] = meta['second_highest_class']
            
            # Marcar si es debate representativo
            if 'representative_debate' in meta:
                debate_info['representative_debate'] = meta['representative_debate']
            
            helpers.save_colored_debate(image, meta['debate_moves'], colored_path, debate_info)
    
    # Guardar datos de confianza si está habilitado
    if args.track_confidence and 'confidence_progression' in meta:
        save_confidence_data(meta['confidence_progression'], i, args, id)

def save_confidence_data(confidence_progression, sample_index, args, id):
    """Guarda los datos de confianza del juez para análisis estadístico."""
    if not confidence_progression:
        return
    
    folder_note = f'_{args.note.replace(" ", "_")}' if args.note and len(args.note) < 20 else ""
    run_folder = f"outputs/debate_{id}{folder_note}"
    os.makedirs(run_folder, exist_ok=True)
    
    confidence_path = os.path.join(run_folder, f"sample_{sample_index}_confidence.json")
    
    # Preparar metadata adicional
    confidence_data = {
        "metadata": {
            "sample_index": sample_index,
            "run_id": str(id),
            "agent_type": args.agent_type if not args.mixed_agents else f"{args.honest_agent}_vs_mixed",
            "precommit": bool(args.precommit),
            "k_pixels": args.k,
            "started_by": args.starts,
            "allow_all_pixels": bool(args.allow_all_pixels),
            "judge_name": args.judge_name,
            "resolution": args.resolution,
            "threshold": args.thr,
            "timestamp": id
        },
        "confidence_progression": confidence_progression
    }
    
    helpers.save_play(confidence_data, confidence_path)
    return confidence_path

def log_results(args, accuracy, id):
    os.makedirs("outputs", exist_ok=True)
    
    if args.mixed_agents:
        # Debates asimétricos - usar archivo específico
        liar_agent_type = 'mcts' if args.honest_agent == 'greedy' else 'greedy'
        
        results = {
            "timestamp": id,
            "judge_name": args.judge_name,
            "seed": args.seed,
            "resolution": args.resolution,
            "thr": args.thr,
            "rollouts": args.rollouts,
            "n_images": args.n_images,
            "pixels": args.k,
            "started": args.starts,
            "precommit": bool(args.precommit),
            "honest_agent_type": args.honest_agent,
            "liar_agent_type": liar_agent_type,
            "allow_all_pixels": bool(args.allow_all_pixels),
            "track_confidence": bool(args.track_confidence),
            "accuracy": accuracy,
            "note": args.note,
        }
        debates_csv_path = "outputs/debates_asimetricos.csv"
        helpers.log_results_csv(debates_csv_path, results)
        print(f"Resultados de debate asimétrico registrados en {debates_csv_path}")
        
    else:
        # Debates simétricos - usar archivo normal
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
            "allow_all_pixels": bool(args.allow_all_pixels),
            "track_confidence": bool(args.track_confidence),
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
    parser.add_argument("--save_colored_debate", action="store_true", help="Guardar imagen coloreada del debate con movimientos numerados")
    parser.add_argument("--mixed_agents", action="store_true", help="Habilitar agentes de tipos diferentes (MCTS vs Greedy)")
    parser.add_argument("--honest_agent", type=str, choices=["greedy", "mcts"], default="greedy", help="Tipo de agente que será honesto (cuando --mixed_agents está activado)")
    parser.add_argument("--precommit", action="store_true")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--starts", type=str, choices=["honest", "liar"], default="liar")
    parser.add_argument("--allow_all_pixels", action="store_true", help="Allow agents to select any pixel, not just relevant ones (>thr)")
    parser.add_argument("--track_confidence", action="store_true", help="Track judge logits/probabilities progression during debate")
    args = parser.parse_args()

    # Generate descriptive experiment ID with timestamp suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    precommit_status = "precommit" if args.precommit else "no-precommit"
    
    if args.mixed_agents:
        # Format: {honest_agent}vs{liar_agent}_{pixels}px_{started}_{precommit_status}_{timestamp}
        liar_agent = "mcts" if args.honest_agent == "greedy" else "greedy"
        id = f"{args.honest_agent}_vs_{liar_agent}_{args.k}px_{args.starts}_{precommit_status}_{timestamp}"
    else:
        # Format: {agent_type}_{pixels}px_{started}_{precommit_status}_{timestamp}
        id = f"{args.agent_type}_{args.k}px_{args.starts}_{precommit_status}_{timestamp}"
    
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
            winner_label, avg_logits, winning_logits = run_exhaustive_precommit(
                image, label_true, args, judge_model, device
            )
            predicted_label = winner_label
            correct += 1 if predicted_label == label_true else 0
            
            # Para precommit, no tenemos movimientos individuales, pero podemos crear metadata básica
            meta = {
                "index": i,
                "truth_label": int(label_true),
                "winner_label": int(winner_label),
                "predicted_label": int(winner_label),
                "avg_logits": {str(lbl): {"true": t, "false": f}
                            for lbl, (t, f) in avg_logits.items()},
                "debate_moves": [],  # Sin movimientos específicos en precommit
                "liar_class": None,  # Será determinado por el ganador si no es la clase real
                "logit_truth": 0.0,  # Se calculará abajo
                "logit_liar": "N/A"  # En precommit exhaustivo, varía por clase
            }
            
            # Usar los logits que realmente determinaron la victoria
            meta["logit_truth"] = winning_logits[0]
            meta["logit_liar"] = winning_logits[1]
            
            # Calcular representante para precommit
            representative_liar_class = None
            
            # Determinar el mentiroso y su clase
            if winner_label != label_true:
                # Mentiroso ganó
                meta["liar_class"] = winner_label
                representative_liar_class = winner_label
            else:
                # Honesto ganó, pero mostrar el mejor mentiroso para contexto
                best_liar_class = None
                best_liar_logit = float('-inf')
                for wrong_lbl, (true_logit, false_logit) in avg_logits.items():
                    if false_logit > best_liar_logit:
                        best_liar_logit = false_logit
                        best_liar_class = wrong_lbl
                if best_liar_class is not None:
                    meta["liar_class"] = best_liar_class
                    meta["logit_liar"] = best_liar_logit
                    representative_liar_class = best_liar_class
            
            # Ejecutar debate representativo solo si necesitamos visualización
            if args.save_colored_debate or args.save_metadata:
                # Usar la clase representativa ya calculada
                if representative_liar_class is None:
                    representative_liar_class = 0  # Fallback
                
                # Ejecutar debate representativo para visualización
                helpers.set_seed(args.seed)  # Semilla consistente para reproducibilidad
                repr_agent_truth, repr_agent_liar = get_agents(
                    args.agent_type, judge_model, label_true, representative_liar_class, 
                    True, image, args.thr, args.rollouts, args.k, args.mixed_agents, args.honest_agent, args.allow_all_pixels
                )
                
                representative_result = run_single_debate(
                    image, repr_agent_truth, repr_agent_liar, args, device
                )
                
                # Usar los movimientos del debate representativo para visualización
                meta["debate_moves"] = representative_result["debate_moves"]
                meta["representative_debate"] = True  # Marcar que es representativo
                
                # Usar la máscara del debate representativo
                representative_mask = representative_result["mask"]
            else:
                # Si no hay visualización, crear máscara vacía
                representative_mask = torch.zeros_like(image)
            
            save_outputs(i, image, representative_mask, meta, args, id)
        else:
            # No precommit, no hay opponent label fijo.
            agent_truth, agent_liar = get_agents(
                args.agent_type, judge_model, label_true, None, args.precommit,
                image, args.thr, args.rollouts, args.k, args.mixed_agents, args.honest_agent, args.allow_all_pixels
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
                "logit_truth": float(res["logit_truth"]),
                "logit_liar": res["logit_liar"],
                "debate_moves": res["debate_moves"],
                "second_highest_logit": res.get("second_highest_logit"),
                "second_highest_class": res.get("second_highest_class")
            }
            save_outputs(i, image, res["mask"], meta, args, id)

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Precisión del juez con debate: {accuracy*100:.2f}% (sobre {total} muestras)")
    log_results(args, accuracy, id)

if __name__ == "__main__":
    main()
