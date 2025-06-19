import argparse
import random
import torch
import numpy as np
from utils import data_utils, helpers
from utils.paths import get_debate_folder, get_model_path, DEBATES_CSV, DEBATES_ASIMETRICOS_CSV
from models.sparse_cnn import SparseCNN
from agents.greedy_agent import GreedyAgent
# from agents.mcts_agent import MCTSAgent
from agents.mcts_fast import FastMCTSAgent as MCTSAgent
from datetime import datetime
from tqdm import tqdm
import os

def load_judge_model(judge_name, resolution, device):
    """Loads judge model from disk."""
    judge_model = SparseCNN(resolution=resolution).to(device)
    model_path = get_model_path(judge_name)
    judge_model.load_state_dict(torch.load(model_path, map_location=device))
    judge_model.eval()
    return judge_model

def get_agents(agent_type, judge_model, label, opponent_label, precommit, image, thr, rollouts, k, mixed_agents=False, honest_agent=None, allow_all_pixels=False):
    """Initializes agents according to type."""
    if mixed_agents:
        # Mixed mode: one MCTS agent and one Greedy
        if honest_agent == "mcts":
            agent_truth = MCTSAgent(judge_model, label, opponent_label, precommit, image, thr, rollouts, k, True, allow_all_pixels)
            agent_liar = GreedyAgent(judge_model, opponent_label, label, precommit, image, thr, allow_all_pixels)
        else:  # honest_agent == "greedy"
            agent_truth = GreedyAgent(judge_model, label, opponent_label, precommit, image, thr, allow_all_pixels)
            agent_liar = MCTSAgent(judge_model, opponent_label, label, precommit, image, thr, rollouts, k, False, allow_all_pixels)
    else:
        # Normal mode: both agents of same type
        if agent_type == "greedy":
            agent_truth = GreedyAgent(judge_model, label, opponent_label, precommit, image, thr, allow_all_pixels)
            agent_liar = GreedyAgent(judge_model, opponent_label, label, precommit, image, thr, allow_all_pixels)
        else:
            agent_truth = MCTSAgent(judge_model, label, opponent_label, precommit, image, thr, rollouts, k, True, allow_all_pixels)
            agent_liar = MCTSAgent(judge_model, opponent_label, label, precommit, image, thr, rollouts, k, False, allow_all_pixels)
    return agent_truth, agent_liar

def run_single_debate(image, agent_truth, agent_liar, args, device):
    """
    Executes a single debate:
    - Alternates turns according to starting agent (args.starts).
    - Each agent plays its move until completing k moves or exhausting options.
    - Returns final mask, logits, predicted_label, etc.
    """
    H, W = image.shape[-2], image.shape[-1]
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Define agent order
    if args.starts == "honest":
        agent_order = [agent_truth, agent_liar]
        agent_names = ["honest", "liar"]
    else:
        agent_order = [agent_liar, agent_truth]
        agent_names = ["liar", "honest"]
    
    # Track moves for colored visualization
    debate_moves = []
    
    # Track judge confidence and logits if enabled
    confidence_progression = []
    should_track = args.track_confidence or args.save_json or args.track_logits or args.track_logits_progressive
    
    if should_track:
        # Evaluate initial state (no pixels revealed)
        values_plane_init = torch.zeros_like(mask)
        judge_input_init = torch.stack([mask, values_plane_init], dim=0).unsqueeze(0)
        with torch.no_grad():
            output_init = agent_truth.judge(judge_input_init)
            logits_init = output_init[0].cpu().numpy()
            probs_init = torch.softmax(output_init[0], dim=0).cpu().numpy()
            entropy_init = -(probs_init * np.log(probs_init + 1e-8)).sum()
            
            # Create judge state for new format
            judge_state = {
                "probabilities": [float(probs_init[i]) for i in range(10)],
                "predicted_class": int(output_init[0].argmax().item()),
                "confidence": float(probs_init.max()),
                "entropy": float(entropy_init)
            }
            
            # Add raw logits if requested
            if args.save_raw_logits:
                judge_state["raw_logits"] = [float(logits_init[i]) for i in range(10)]
            
            confidence_progression.append({
                "turn": 0,
                "pixels_revealed": 0,
                "agent": "initial",
                "pixel_revealed": None,
                "judge_state": judge_state,
                # Legacy format for backward compatibility
                "logits": {str(i): float(logits_init[i]) for i in range(10)},
                "probabilities": {str(i): float(probs_init[i]) for i in range(10)},
                "predicted_class": int(output_init[0].argmax().item()),
                "max_confidence": float(probs_init.max())
            })
    
    # Alternate turns
    for move in range(args.k):
        current_agent_idx = move % 2
        current_agent = agent_order[current_agent_idx]
        current_agent_name = agent_names[current_agent_idx]
        
        pixel = current_agent.choose_pixel(mask, reveal_count=move)
        if pixel is None:
            break
        y, x = pixel
        mask[y, x] = 1.0
        
        # Save move for visualization
        debate_moves.append((y, x, current_agent_name, move + 1))
        
        # Track judge confidence and logits after this move
        if should_track and (args.track_logits_progressive or args.track_confidence or args.save_json):
            values_plane_current = image * mask
            judge_input_current = torch.stack([mask, values_plane_current], dim=0).unsqueeze(0)
            with torch.no_grad():
                output_current = agent_truth.judge(judge_input_current)
                logits_current = output_current[0].cpu().numpy()
                probs_current = torch.softmax(output_current[0], dim=0).cpu().numpy()
                entropy_current = -(probs_current * np.log(probs_current + 1e-8)).sum()
                
                # Create judge state for new format
                judge_state = {
                    "probabilities": [float(probs_current[i]) for i in range(10)],
                    "predicted_class": int(output_current[0].argmax().item()),
                    "confidence": float(probs_current.max()),
                    "entropy": float(entropy_current)
                }
                
                # Add raw logits if requested
                if args.save_raw_logits:
                    judge_state["raw_logits"] = [float(logits_current[i]) for i in range(10)]
                
                confidence_progression.append({
                    "turn": move + 1,
                    "pixels_revealed": int(mask.sum().item()),
                    "agent": current_agent_name,
                    "pixel_revealed": (int(y), int(x)),
                    "judge_state": judge_state,
                    # Legacy format for backward compatibility
                    "logits": {str(i): float(logits_current[i]) for i in range(10)},
                    "probabilities": {str(i): float(probs_current[i]) for i in range(10)},
                    "predicted_class": int(output_current[0].argmax().item()),
                    "max_confidence": float(probs_current.max())
                })

    # Calculate judge output
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

    # For no-precommit mode, calculate second highest logit
    second_highest_logit = None
    second_highest_class = None
    if not args.precommit:
        logits_sorted = output[0].cpu().numpy()
        sorted_indices = np.argsort(logits_sorted)[::-1]  # Orden descendente
        if len(sorted_indices) >= 2:
            second_highest_class = int(sorted_indices[1])  # Convertir a int
            second_highest_logit = float(logits_sorted[second_highest_class])  # Convertir a float

    # Final confidence and logits tracking
    if should_track and (args.track_logits or args.track_logits_progressive or args.track_confidence or args.save_json):
        logits_final = output[0].cpu().numpy()
        probs_final = torch.softmax(output[0], dim=0).cpu().numpy()
        entropy_final = -(probs_final * np.log(probs_final + 1e-8)).sum()
        
        # Create judge state for final evaluation
        judge_state_final = {
            "probabilities": [float(probs_final[i]) for i in range(10)],
            "predicted_class": predicted_label,
            "confidence": float(probs_final.max()),
            "entropy": float(entropy_final)
        }
        
        # Add raw logits if requested
        if args.save_raw_logits:
            judge_state_final["raw_logits"] = [float(logits_final[i]) for i in range(10)]
        
        confidence_progression.append({
            "turn": args.k + 1,
            "pixels_revealed": int(mask.sum().item()),
            "agent": "final",
            "pixel_revealed": None,
            "judge_state": judge_state_final,
            # Legacy format for backward compatibility
            "logits": {str(i): float(logits_final[i]) for i in range(10)},
            "probabilities": {str(i): float(probs_final[i]) for i in range(10)},
            "predicted_class": predicted_label,
            "max_confidence": float(probs_final.max())
        })

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
    
    # Agregar progresión de confianza y resumen de logits si está habilitada
    if should_track:
        result["confidence_progression"] = confidence_progression
        
        # Add logits summary for quick access
        if confidence_progression and any("judge_state" in entry for entry in confidence_progression):
            judge_states = [entry["judge_state"] for entry in confidence_progression if "judge_state" in entry]
            if judge_states:
                result["logits_summary"] = {
                    "initial_confidence": judge_states[0]["confidence"] if len(judge_states) > 0 else None,
                    "final_confidence": judge_states[-1]["confidence"] if len(judge_states) > 0 else None,
                    "confidence_evolution": [state["confidence"] for state in judge_states],
                    "entropy_evolution": [state["entropy"] for state in judge_states]
                }
    
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
      - representative_debate_data: logits tracking data from the winning debate (if tracking enabled)
    """
    logits_avg_dct = {}
    winner_label = label_true  # Por defecto gana el honesto
    winning_logits = (0.0, 0.0)  # Default para cuando gana el honesto
    representative_debate_data = None  # Store logits data from representative debate

    wrong_labels = [lbl for lbl in range(10) if lbl != label_true]

    for wrong_lbl in wrong_labels:
        true_accum, false_accum = 0.0, 0.0
        debate_results = []  # Store results for each seed
        
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
            debate_results.append(res)

        # Promediar
        avg_true  = true_accum  / 3.0
        avg_false = false_accum / 3.0
        logits_avg_dct[wrong_lbl] = (avg_true, avg_false)

        # Si el mentiroso supera al honesto EN ALGUNA etiqueta -> mentiroso gana
        if avg_false > avg_true:
            winner_label = wrong_lbl
            winning_logits = (avg_true, avg_false)  # Guardar los logits que determinaron la victoria
            # Store representative debate data (from first seed of winning scenario)
            if debate_results and (args.track_logits or args.track_logits_progressive or args.save_json):
                representative_debate_data = debate_results[0]
            break  # basta con una etiqueta para que pierda la verdad

    # Si ningún mentiroso ganó, usar los logits del honesto vs el mejor mentiroso
    if winner_label == label_true and logits_avg_dct:
        # Encontrar el mejor desafío del mentiroso
        best_liar_class = max(logits_avg_dct.keys(), key=lambda k: logits_avg_dct[k][1])
        winning_logits = logits_avg_dct[best_liar_class]
        
        # For honest winner, get representative data from the best challenge
        if args.track_logits or args.track_logits_progressive or args.save_json:
            # Re-run the best challenge with first seed to get logits data
            helpers.set_seed(args.seed)
            agent_truth, agent_liar = get_agents(
                args.agent_type, judge_model, label_true, best_liar_class, args.precommit, image, args.thr, args.rollouts, args.k,
                args.mixed_agents, args.honest_agent, args.allow_all_pixels
            )
            representative_debate_data = run_single_debate(image, agent_truth, agent_liar, args, device)

    return winner_label, logits_avg_dct, winning_logits, representative_debate_data

def save_outputs(i, image, mask, meta, args, id):
    """Saves image, mask and metadata to output directory."""
    if args.save_mask or args.save_play or args.save_images or args.save_metadata or args.save_colored_debate or args.track_confidence or args.save_json:
        folder_note = args.note.replace(" ", "_") if args.note and len(args.note) < 20 else ""
        run_folder = get_debate_folder(id, folder_note)
    
    if args.save_images or args.save_metadata:
        img_path = os.path.join(run_folder, f"sample_{i}_image.png")
        helpers.save_image(image, img_path)   
         
    if args.save_mask or args.save_metadata:
        img_path = os.path.join(run_folder, f"sample_{i}_mask.png")
        helpers.save_mask(image, mask, img_path)
        
    # Only save legacy play.json if save_json is not enabled (avoid redundancy)
    if (args.save_play or args.save_metadata) and not args.save_json:
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
    
    # Guardar datos de confianza si está habilitado (para análisis estadístico separado)
    if args.track_confidence and 'confidence_progression' in meta:
        save_confidence_data(meta['confidence_progression'], i, args, id)
    
    # Guardar JSON comprehensivo si está habilitado
    if args.save_json:
        json_path = os.path.join(run_folder, f"sample_{i}_comprehensive.json")
        # Add original image to meta for pixel value extraction
        meta['original_image'] = image
        comprehensive_data = create_comprehensive_json(i, meta.get('truth_label', 0), meta, args, id, image.shape)
        helpers.save_comprehensive_json(comprehensive_data, json_path)

def save_confidence_data(confidence_progression, sample_index, args, id):
    """
    Guarda los datos de progresión de probabilidades del juez para análisis estadístico.
    Se enfoca en probabilidades (softmax) para análisis de confianza.
    """
    if not confidence_progression:
        return
    
    folder_note = f'_{args.note.replace(" ", "_")}' if args.note and len(args.note) < 20 else ""
    run_folder = f"outputs/debate_{id}{folder_note}"
    os.makedirs(run_folder, exist_ok=True)
    
    confidence_path = os.path.join(run_folder, f"sample_{sample_index}_probabilities.json")
    
    # Extraer solo datos relevantes para análisis estadístico (enfoque en probabilidades)
    probability_progression = []
    for step in confidence_progression:
        probability_step = {
            "turn": step["turn"],
            "pixels_revealed": step["pixels_revealed"],
            "agent": step["agent"],
            "pixel_revealed": step["pixel_revealed"],
            "probabilities": step["probabilities"],  # Solo probabilidades para análisis
            "predicted_class": step["predicted_class"],
            "max_confidence": step["max_confidence"]
        }
        probability_progression.append(probability_step)
    
    # Metadata específica para análisis de probabilidades
    probability_data = {
        "analysis_type": "judge_probability_progression",
        "description": "Statistical analysis data focusing on judge probability evolution during debate",
        "metadata": {
            "sample_index": sample_index,
            "run_id": str(id),
            "agent_config": args.agent_type if not args.mixed_agents else f"{args.honest_agent}_vs_mixed",
            "precommit": bool(args.precommit),
            "k_pixels": args.k,
            "started_by": args.starts,
            "allow_all_pixels": bool(args.allow_all_pixels),
            "judge_name": args.judge_name,
            "resolution": args.resolution,
            "threshold": args.thr,
            "timestamp": id
        },
        "probability_progression": probability_progression
    }
    
    helpers.save_play(probability_data, confidence_path)
    return confidence_path

def create_comprehensive_json(i, label_true, result_data, args, id, image_shape):
    """
    Creates comprehensive JSON metadata in the new format.
    Combines functionality of save_play and track_confidence.
    """
    from datetime import datetime
    
    # Extract data from result_data
    predicted_label = result_data.get("predicted_label", 0)
    logits = result_data.get("logits", {})
    confidence_progression = result_data.get("confidence_progression", [])
    debate_moves = result_data.get("debate_moves", [])
    
    # Convert logits to dict format if it's a numpy array
    if hasattr(logits, '__len__') and not isinstance(logits, dict):
        logits_dict = {str(i): float(logits[i]) for i in range(len(logits))}
    elif isinstance(logits, dict):
        logits_dict = {str(k): float(v) for k, v in logits.items()}
    else:
        logits_dict = {}
    
    # Find closest runner-up (second highest logit)
    if logits_dict:
        sorted_logits = sorted(logits_dict.items(), key=lambda x: x[1], reverse=True)
        closest_runner_up = {
            "class": int(sorted_logits[1][0]) if len(sorted_logits) > 1 else None,
            "logit": float(sorted_logits[1][1]) if len(sorted_logits) > 1 else None
        }
    else:
        closest_runner_up = {"class": None, "logit": None}
    
    # Build debate moves with pixel values (get pixel values from original image)
    processed_moves = []
    for move in debate_moves:
        if len(move) >= 4:  # (y, x, agent, turn)
            y, x, agent, turn = move[:4]
            # Get pixel value from result_data if available
            pixel_value = None
            if 'original_image' in result_data:
                # Get pixel value from original image tensor
                try:
                    original_image = result_data['original_image']
                    if torch.is_tensor(original_image):
                        if original_image.dim() == 3:  # 1xHxW
                            pixel_value = float(original_image[0, y, x].item())
                        else:  # HxW
                            pixel_value = float(original_image[y, x].item())
                    else:
                        pixel_value = float(original_image[y, x])
                except (IndexError, TypeError):
                    pixel_value = None
            
            processed_moves.append({
                "turn": int(turn),
                "agent": str(agent),
                "pixel_position": [int(y), int(x)],
                "pixel_value": pixel_value
            })
    
    # Create comprehensive metadata
    comprehensive_data = {
        "metadata": {
            "run_id": str(id),
            "sample_index": int(i),
            "timestamp": datetime.now().isoformat(),
            "experiment_config": {
                "judge_name": str(args.judge_name),
                "resolution": int(args.resolution),
                "threshold": float(args.thr),
                "seed": int(args.seed),
                "k_pixels": int(args.k),
                "started_by": str(args.starts),
                "precommit": bool(args.precommit),
                "allow_all_pixels": bool(args.allow_all_pixels),
                "mixed_agents": bool(args.mixed_agents),
                "agent_type": str(args.agent_type) if not args.mixed_agents else None,
                "honest_agent_type": str(args.honest_agent) if args.mixed_agents else str(args.agent_type),
                "liar_agent_type": ('mcts' if args.honest_agent == 'greedy' else 'greedy') if args.mixed_agents else str(args.agent_type),
                "rollouts": int(args.rollouts) if args.agent_type == "mcts" else 0
            }
        },
        "ground_truth": {
            "true_class": int(label_true),
            "image_index": int(i)
        },
        "final_result": {
            "predicted_class": int(predicted_label),
            "is_correct": bool(predicted_label == label_true),
            "final_logits": logits_dict,
            "closest_runner_up": closest_runner_up
        },
        "debate_reconstruction": {
            "total_moves": len(processed_moves),
            "pixels_revealed": int(result_data.get("pixels_revealed", len(processed_moves))),
            "moves": processed_moves
        }
    }
    
    # Add precommit information if applicable
    if args.precommit:
        liar_class = result_data.get("liar_class")
        liar_logit = result_data.get("logit_liar")
        honest_logit = result_data.get("logit_truth")
        
        comprehensive_data["precommit_info"] = {
            "liar_committed_class": int(liar_class) if liar_class is not None else None,
            "liar_final_logit": float(liar_logit) if liar_logit != "N/A" and liar_logit is not None else None,
            "honest_final_logit": float(honest_logit) if honest_logit is not None else None
        }
    
    # Add confidence progression if available
    if confidence_progression:
        comprehensive_data["confidence_progression"] = confidence_progression
    
    # Add logits summary if available
    logits_summary = result_data.get("logits_summary", {})
    if logits_summary:
        comprehensive_data["logits_summary"] = logits_summary
    
    # Integrate judge_state into moves if progressive tracking is enabled
    if confidence_progression and (args.track_logits_progressive or args.save_json):
        # Create a mapping from turn to judge_state
        judge_state_map = {}
        for entry in confidence_progression:
            if "judge_state" in entry and entry.get("turn", 0) > 0:
                judge_state_map[entry["turn"]] = entry["judge_state"]
        
        # Add judge_state to moves
        for move in comprehensive_data["debate_reconstruction"]["moves"]:
            turn = move["turn"]
            if turn in judge_state_map:
                move["judge_state"] = judge_state_map[turn]
    
    return comprehensive_data

def log_results(args, accuracy, id):
    if args.mixed_agents:
        # Asymmetric debates - use specific file
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
        helpers.log_results_csv(str(DEBATES_ASIMETRICOS_CSV), results)
        print(f"Asymmetric debate results logged to {DEBATES_ASIMETRICOS_CSV}")
        
    else:
        # Symmetric debates - use normal file
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
        helpers.log_results_csv(str(DEBATES_CSV), results)
        print(f"Results logged to {DEBATES_CSV}")

def main():
    parser = argparse.ArgumentParser(description="Run AI Safety via Debate experiment on MNIST")
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
    parser.add_argument("--save_play", action="store_true", help="[DEPRECATED] Save play metadata in JSON format. Use --save_json instead.")
    parser.add_argument("--save_metadata", action="store_true", help="Save metadata (images, masks and plays) for each debate")
    parser.add_argument("--save_colored_debate", action="store_true", help="Save colored debate image with numbered moves")
    parser.add_argument("--mixed_agents", action="store_true", help="Enable different agent types (MCTS vs Greedy)")
    parser.add_argument("--honest_agent", type=str, choices=["greedy", "mcts"], default="greedy", help="Type of agent that will be honest (when --mixed_agents is enabled)")
    parser.add_argument("--precommit", action="store_true")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--starts", type=str, choices=["honest", "liar"], default="liar")
    parser.add_argument("--allow_all_pixels", action="store_true", help="Allow agents to select any pixel, not just relevant ones (>thr)")
    parser.add_argument("--track_confidence", action="store_true", help="Track judge probabilities progression for statistical analysis (creates separate confidence file)")
    parser.add_argument("--track_logits", action="store_true", help="Track judge logits at final state only")
    parser.add_argument("--track_logits_progressive", action="store_true", help="Track judge logits after each pixel selection")
    parser.add_argument("--save_raw_logits", action="store_true", help="Save raw logits in addition to softmax probabilities")
    parser.add_argument("--save_json", action="store_true", help="Save comprehensive JSON metadata (replaces save_play, includes confidence_progression)")
    args = parser.parse_args()
    
    # Issue deprecation warnings
    if args.save_play and not args.save_json:
        print("WARNING: --save_play is deprecated. Use --save_json instead for comprehensive metadata.")
    elif args.save_play and args.save_json:
        print("INFO: --save_play is redundant when using --save_json. Consider removing --save_play.")
    
    # Enable confidence tracking if logits tracking is requested
    if args.track_logits or args.track_logits_progressive:
        args.track_confidence = True
        print("INFO: Confidence tracking automatically enabled for logits tracking.")
    
    # Auto-enable comprehensive JSON when save_metadata is used to avoid redundant files
    if args.save_metadata and not args.save_json:
        args.save_json = True
        print("INFO: Comprehensive JSON automatically enabled with --save_metadata to provide complete data in one file.")
    
    # Info message for track_confidence usage
    if args.track_confidence:
        print("INFO: --track_confidence will create separate probability analysis files for statistical research.")

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
            winner_label, avg_logits, winning_logits, representative_debate = run_exhaustive_precommit(
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
            
            # Add logits tracking data from representative debate if available
            if representative_debate and (args.track_logits or args.track_logits_progressive or args.save_json):
                meta["confidence_progression"] = representative_debate.get("confidence_progression", [])
                meta["logits_summary"] = representative_debate.get("logits_summary", {})
                # Update final logits from representative debate
                if "logits" in representative_debate:
                    meta["logits"] = {str(i): float(representative_debate["logits"][i]) for i in range(10)}
            
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
                "second_highest_class": res.get("second_highest_class"),
                "confidence_progression": res.get("confidence_progression", []),
                "logits_summary": res.get("logits_summary", {})
            }
            save_outputs(i, image, res["mask"], meta, args, id)

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Judge accuracy with debate: {accuracy*100:.2f}% (on {total} samples)")
    log_results(args, accuracy, id)

if __name__ == "__main__":
    main()
