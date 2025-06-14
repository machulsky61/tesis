import os
import csv
import json
import random
import numpy as np
import torch
from datetime import datetime

def load_times_font(size=18, bold=False):
    # List of likely paths
    candidates = [
        "C:/Windows/Fonts/timesnewroman.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "times.ttf"
    ]
    
    if bold:
        candidates = [path.replace("times", "timesbd") for path in candidates]

    for path in candidates:
        if os.path.exists(path):
            from PIL import ImageFont
            return ImageFont.truetype(path, size)
    
    from PIL import ImageFont
    return ImageFont.load_default()  # fallback

def set_seed(seed):
    """Sets global seed for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using GPU, also set seed for CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Options for CNN reproducibility (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_image(original_image, filepath):
    """
    Guarda una imagen PNG de original_image.
    original_image puede ser un Tensor 1xHxW o HxW, o un arreglo numpy 2D.
    """
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Move data to CPU and convert to numpy array
    if torch.is_tensor(original_image):
        img = original_image.clone().detach().cpu()
        # If has initial channel (1xHxW), convert to 2D HxW
        if img.dim() == 3:
            img = img[0]
    else:
        # Assume original_image is already 2D numpy array
        img = torch.tensor(original_image)
    # Convert to 0-255 uint8 scale for saving
    img_np = (img.numpy() * 255).astype(np.uint8)
    # Use PIL to save in grayscale mode
    try:
        from PIL import Image
    except ImportError:
        # If PIL not available, try with OpenCV (cv2)
        try:
            import cv2
            cv2.imwrite(filepath, img_np)
            return
        except ImportError:
            raise RuntimeError("Neither PIL nor cv2 found for saving images")
    img_pil = Image.fromarray(img_np, mode='L')
    img_pil.save(filepath)

def save_mask(original_image, mask, filepath):
    """
    Saves a PNG image showing revealed pixels of original_image.
    Revealed pixels retain their original value; unrevealed appear black.
    """
    # Ensure destination folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Move data to CPU and convert to numpy array
    # original_image puede ser Tensor 1xHxW o HxW; mask puede ser Tensor HxW (o 1xHxW)
    if torch.is_tensor(original_image):
        img = original_image.clone().detach().cpu()
        # If has initial channel (1xHxW), convert to 2D HxW
        if img.dim() == 3:
            img = img[0]
    else:
        # Assume original_image is already 2D numpy array
        img = torch.tensor(original_image)
    if torch.is_tensor(mask):
        msk = mask.clone().detach().cpu()
        if msk.dim() == 3:
            msk = msk[0]
    else:
        msk = torch.tensor(mask)
    # Asegurar que img y msk son float 
    img = img.float()
    msk = msk.float()
    
    
    # Aplicar máscara: píxeles no revelados a 0
    # Si la máscara es binaria (0s y 1s), usarla como está
    # Si la máscara tiene valores intermedios, normalizarla
    if msk.max() <= 1.0:
        revealed_img = img * msk
    else:
        # Si la máscara tiene valores > 1, normalizarla primero
        msk_normalized = msk / msk.max()
        revealed_img = img * msk_normalized
    # Convert to 0-255 uint8 scale for saving
    # Suponemos original_image normalizada 0-1 (como ToTensor de MNIST)
    revealed_np = (revealed_img.numpy() * 255).astype(np.uint8)
    # Use PIL to save in grayscale mode
    try:
        from PIL import Image
    except ImportError:
        # If PIL not available, try with OpenCV (cv2)
        try:
            import cv2
            cv2.imwrite(filepath, revealed_np)
            return
        except ImportError:
            raise RuntimeError("Neither PIL nor cv2 found for saving images")
    img_pil = Image.fromarray(revealed_np, mode='L')
    img_pil.save(filepath)

def save_play(metadata, filepath):
    """
    Saves debate metadata (e.g., revelation sequence, classes, logits, etc.) to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def save_colored_debate(original_image, debate_moves, filepath, debate_info=None):
    """
    Saves a PNG image showing debate pixels colored by agent and numbered by order,
    with detailed debate information.
    
    Args:
        original_image: imagen original (Tensor HxW o 1xHxW)
        debate_moves: lista de movimientos [(y, x, agent_type, move_number), ...]
                     donde agent_type es 'honest' o 'liar'
        filepath: ruta donde guardar la imagen
        debate_info: diccionario con información del debate {
            'debate_id': str, 'run_id': str, 'sample_index': int,
            'true_label': int, 'liar_label': int (opcional),
            'predicted_label': int, 'predicted_logit': float,
            'honest_logit': float, 'liar_logit': float (opcional),
            'honest_agent_type': str, 'liar_agent_type': str,
            'agent_types': str (descripción general)
        }
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Generar texto informativo con estilo prolijo
    def generate_info_text():
        if not debate_info:
            return []
        
        
        info_lines = []
        
        # Titulo general (sin run ID)
        info_lines.append("Debate Simulation")
        info_lines.append("")
        
        # Configuracion
        honest_type = debate_info.get('honest_agent_type', 'unknown').title()
        liar_type = debate_info.get('liar_agent_type', 'unknown').title()
        sample_idx = debate_info.get('sample_index', '?')
        
        # Determine if precommit
        has_precommit = ('liar_label' in debate_info and debate_info['liar_label'] is not None) or \
                       debate_info.get('representative_debate', False)
        
        # First move
        first_move = debate_info.get('first_move', 'liar')
        first_move_text = "Blue Player" if first_move == "honest" else "Red Player"
        
        info_lines.append("Configuration:")
        info_lines.append(f"  - Agents: {honest_type} (Blue) vs {liar_type} (Red)")
        info_lines.append(f"  - Precommit: {'Enabled' if has_precommit else 'Disabled'}")
        info_lines.append(f"  - First move: {first_move_text}")
        info_lines.append(f"  - Sample: #{sample_idx}")
        info_lines.append("")
        
        # Resultado
        true_label = debate_info.get('true_label', '?')
        pred_label = debate_info.get('predicted_label', '?')
        pred_logit = debate_info.get('predicted_logit', 0.0)
        
        if str(true_label) == str(pred_label):
            result_symbol = "Correct"
        else:
            result_symbol = "Incorrect"
        
        info_lines.append("Outcome:")
        info_lines.append(f"  - True label: {true_label} ")
        info_lines.append(f"  - Predicted: {pred_label} (Logit: {pred_logit:.2f}) → {result_symbol}")
        info_lines.append("")
        
        # Blue Player (Honest)
        honest_logit = debate_info.get('honest_logit', 0.0)
        info_lines.append("Blue Player (Honest):")
        info_lines.append(f"  - Agent: {honest_type}")
        info_lines.append(f"  - Target: Class {true_label} (Logit: {honest_logit:.2f})")
        info_lines.append("")
        
        # Red Player (Liar)
        info_lines.append("Red Player (Liar):")
        info_lines.append(f"  - Agent: {liar_type}")
        
        if has_precommit and 'liar_label' in debate_info:
            liar_label = debate_info['liar_label']
            liar_logit = debate_info.get('liar_logit', 0.0)
            info_lines.append(f"  - Target: Class {liar_label} (Logit: {liar_logit:.2f})")
        else:
            info_lines.append("  - Target: Not fixed")
            
            # Para no-precommit, agregar segundo logit más alto
            second_highest_logit = debate_info.get('second_highest_logit', None)
            second_highest_class = debate_info.get('second_highest_class', None)
            if second_highest_logit is not None and second_highest_class is not None:
                info_lines.append(f"  - Runner-up: Class {second_highest_class} (Logit: {second_highest_logit:.2f})")
        
        info_lines.append("")
        
        # Run ID al final en gris pequeño
        run_id = debate_info.get('run_id', '?????')
        info_lines.append(f"Run ID: {run_id}")
        
        return info_lines
    
    info_text = generate_info_text()
    
    # Convertir imagen a numpy array
    if torch.is_tensor(original_image):
        img = original_image.clone().detach().cpu()
        if img.dim() == 3:
            img = img[0]
    else:
        img = torch.tensor(original_image)
    
    img_np = (img.numpy() * 255).astype(np.uint8)
    H, W = img_np.shape
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Escalar la imagen para mejor visualización (imagen de alta resolución)
        scale_factor = max(1, 800 // max(H, W))  # Escalar para que sea al menos 800px
        scaled_H, scaled_W = H * scale_factor, W * scale_factor
        
        # Calcular espacio para información (panel lateral) - aumentado para fuentes más grandes
        info_panel_width = 600 if info_text else 0  # Más ancho para fuentes más grandes
        total_width = scaled_W + info_panel_width
        total_height = max(scaled_H, len(info_text) * 30 + 100) if info_text else scaled_H  # Más espacio para fuentes grandes
        
        # Crear imagen completa con espacio para información
        full_img = Image.new('RGB', (total_width, total_height), color=(240, 240, 240))
        
        # Convertir imagen original a RGB y escalar
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
        img_pil = Image.fromarray(img_rgb, mode='RGB')
        img_pil = img_pil.resize((scaled_W, scaled_H), Image.NEAREST)  # NEAREST para mantener píxeles nítidos
        
        # Pegar la imagen escalada en la imagen completa
        full_img.paste(img_pil, (0, 0))
        
        # Dibujar sobre la imagen completa
        draw = ImageDraw.Draw(full_img)
        
        # Colores para los agentes
        honest_color = (0, 100, 255)    # Azul
        liar_color = (255, 50, 50)      # Rojo
        
        # Dibujar píxeles coloreados y números
        for y, x, agent_type, move_number in debate_moves:
            color = honest_color if agent_type == 'honest' else liar_color
            
            # Coordenadas escaladas - cubrir todo el pixel escalado
            scaled_x_start = x * scale_factor
            scaled_y_start = y * scale_factor
            scaled_x_end = (x + 1) * scale_factor - 1
            scaled_y_end = (y + 1) * scale_factor - 1
            
            # Colorear todo el pixel escalado
            draw.rectangle([
                scaled_x_start, scaled_y_start, 
                scaled_x_end, scaled_y_end
            ], fill=color)
            
            # Agregar número del movimiento (más pequeño)
            if scale_factor >= 6:  # Solo mostrar números si hay suficiente espacio
                try:
                    text = str(move_number)
                    
                    # Calcular tamaño de fuente proporcional al pixel escalado
                    try:
                        # Tamaño de fuente = 70% del tamaño del pixel escalado (mínimo 20px)
                        font_size = max(20, int(scale_factor * 0.7))
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        # Fallback a fuente por defecto
                        font = ImageFont.load_default()
                    
                    # Calcular posición del texto perfectamente centrado
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Centrar perfectamente en el pixel escalado
                    text_x = scaled_x_start + (scale_factor - text_width) // 2
                    text_y = scaled_y_start + (scale_factor - text_height) // 2
                    
                    # Ajuste fino para centrado perfecto
                    text_x = max(scaled_x_start + 1, text_x)
                    text_y = max(scaled_y_start + 1, text_y)
                    
                    # Dibujar texto con contorno blanco para mejor visibilidad
                    outline_color = (255, 255, 255)
                    text_color = (0, 0, 0)
                    
                    # Contorno más delgado para números pequeños
                    for offset_x in [-1, 0, 1]:
                        for offset_y in [-1, 0, 1]:
                            if offset_x != 0 or offset_y != 0:
                                draw.text((text_x + offset_x, text_y + offset_y), text, 
                                        fill=outline_color, font=font)
                    
                    # Texto principal
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
                    
                except Exception as e:
                    # Si falla el texto, solo colorear el pixel
                    print(f"Warning: No se pudo dibujar el número {move_number}: {e}")
                    pass
        
        # Dibujar panel de información
        if info_text and info_panel_width > 0:
            info_x = scaled_W + 25  # Margen desde la imagen
            info_y = 25
            
            # Cargar fuentes Times New Roman con función robusta (tamaños aumentados)
            title_font = load_times_font(28, bold=True)     # Título en negrita
            header_font = load_times_font(24, bold=True)    # Headers en negrita
            content_font = load_times_font(20, bold=False)  # Contenido normal
            
            # Dibujar cada línea con formato apropiado (todo en negro)
            for i, line in enumerate(info_text):
                if line == "":  # Blank line
                    info_y += 18
                elif line.startswith("Debate Simulation"):
                    # Main title
                    draw.text((info_x, info_y), line, fill=(0, 0, 0), font=title_font)
                    info_y += 40
                elif line.startswith("Configuration:") or line.startswith("Outcome:") or \
                     line.startswith("Blue Player") or line.startswith("Red Player"):
                    # Section headers
                    draw.text((info_x, info_y), line, fill=(0, 0, 0), font=header_font)
                    info_y += 32
                elif line.startswith("  -"):
                    # Indented details
                    draw.text((info_x, info_y), line, fill=(0, 0, 0), font=content_font)
                    info_y += 26
                elif line.startswith("Run ID:"):
                    # Run ID en gris y más pequeño
                    small_font = load_times_font(16, bold=False)  # También aumentado ligeramente
                    draw.text((info_x, info_y), line, fill=(120, 120, 120), font=small_font)
                    info_y += 22
                else:
                    # Normal text
                    draw.text((info_x, info_y), line, fill=(0, 0, 0), font=content_font)
                    info_y += 28
        
        full_img.save(filepath)
        
    except ImportError:
        # Fallback usando OpenCV con escalado
        try:
            import cv2
            
            # Escalar la imagen (imagen de alta resolución)
            scale_factor = max(1, 800 // max(H, W))
            new_H, new_W = H * scale_factor, W * scale_factor
            
            # Convertir a BGR y escalar
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            img_bgr = cv2.resize(img_bgr, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            
            for y, x, agent_type, move_number in debate_moves:
                color = (255, 100, 0) if agent_type == 'honest' else (50, 50, 255)  # BGR format
                
                # Coordenadas escaladas - cubrir todo el pixel
                scaled_x_start = x * scale_factor
                scaled_y_start = y * scale_factor
                scaled_x_end = (x + 1) * scale_factor
                scaled_y_end = (y + 1) * scale_factor
                
                # Colorear todo el pixel escalado
                cv2.rectangle(img_bgr, (scaled_x_start, scaled_y_start), (scaled_x_end-1, scaled_y_end-1), color, -1)
                
                # Agregar número perfectamente centrado
                if scale_factor >= 6:
                    # Calcular escala de fuente = 6% del tamaño del pixel (mínimo 0.8, máximo 1.4)
                    font_scale = max(0.8, min(1.4, scale_factor * 0.06))
                    
                    # Calcular tamaño del texto para centrado perfecto
                    text_size = cv2.getTextSize(str(move_number), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    text_width, text_height = text_size
                    
                    # Centrar perfectamente el texto
                    text_x = scaled_x_start + (scale_factor - text_width) // 2
                    text_y = scaled_y_start + (scale_factor + text_height) // 2
                    
                    # Contorno blanco más delgado
                    cv2.putText(img_bgr, str(move_number), (text_x-1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    cv2.putText(img_bgr, str(move_number), (text_x+1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    cv2.putText(img_bgr, str(move_number), (text_x, text_y-1), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    cv2.putText(img_bgr, str(move_number), (text_x, text_y+1), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    
                    # Texto principal negro
                    cv2.putText(img_bgr, str(move_number), (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
            
            cv2.imwrite(filepath, img_bgr)
            
        except ImportError:
            raise RuntimeError("No se encontró PIL ni cv2 para guardar imágenes coloreadas")

def log_results_csv(logfile, results):
    """
    Logs experiment results to a CSV file with unified format.
    `results` must be a dictionary containing expected keys.
    """
    # Determine log type based on filename
    if "judges.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "epochs","batch_size","lr","best_loss", "pixels", "accuracy","note"]
    elif "evaluations.csv" in logfile:
        columns = ["timestamp","judge_name","strategy","resolution","thr","seed", "n_images", "pixels", "accuracy","rollouts","allow_all_pixels","note"]
    elif "debates_asimetricos.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "rollouts","n_images", "pixels", "started","precommit", 
                  "honest_agent_type","liar_agent_type","allow_all_pixels","track_confidence","accuracy","note"]
    elif "debates.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "rollouts","n_images","agent_type", "pixels", "started","precommit","allow_all_pixels","track_confidence", "accuracy","note"]
    else:
        # Default columns for backward compatibility
        columns = ["timestamp","judge_name","resolution","thr","seed", "pixels", "accuracy","note"]

    # If file doesn't exist or is empty, write header
    file_exists = os.path.isfile(logfile)
    write_header = not file_exists or os.path.getsize(logfile) == 0

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if write_header:
            writer.writeheader()
        # Ensure results contains all columns (fill if any non-essential missing)
        for col in columns:
            if col not in results:
                results[col] = ""
        writer.writerow(results)
