import os
import csv
import json
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed):
    """Fija la semilla global para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Si se utiliza GPU, fijamos semilla para CUDA también
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Opciones para reproducibilidad en CNN (puede afectar rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_image(original_image, filepath):
    """
    Guarda una imagen PNG de original_image.
    original_image puede ser un Tensor 1xHxW o HxW, o un arreglo numpy 2D.
    """
    # Asegurarse de que la carpeta de destino existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Mover datos a CPU y convertir a arreglo numpy
    if torch.is_tensor(original_image):
        img = original_image.clone().detach().cpu()
        # Si tiene canal inicial (1xHxW), convertir a 2D HxW
        if img.dim() == 3:
            img = img[0]
    else:
        # Asumimos que original_image es numpy array 2D ya
        img = torch.tensor(original_image)
    # Convertir a escala de 0-255 uint8 para guardar
    img_np = (img.numpy() * 255).astype(np.uint8)
    # Utilizar PIL para guardar en modo escala de grises
    try:
        from PIL import Image
    except ImportError:
        # Si PIL no está disponible, intentar con OpenCV (cv2)
        try:
            import cv2
            cv2.imwrite(filepath, img_np)
            return
        except ImportError:
            raise RuntimeError("No se encontró PIL ni cv2 para guardar imágenes")
    img_pil = Image.fromarray(img_np, mode='L')
    img_pil.save(filepath)

def save_mask(original_image, mask, filepath):
    """
    Guarda una imagen PNG que muestra los píxeles revelados de original_image.
    Los píxeles revelados conservan su valor original; los no revelados aparecen en negro.
    """
    # Asegurarse de que la carpeta de destino existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Mover datos a CPU y convertir a arreglo numpy
    # original_image puede ser Tensor 1xHxW o HxW; mask puede ser Tensor HxW (o 1xHxW)
    if torch.is_tensor(original_image):
        img = original_image.clone().detach().cpu()
        # Si tiene canal inicial (1xHxW), convertir a 2D HxW
        if img.dim() == 3:
            img = img[0]
    else:
        # Asumimos que original_image es numpy array 2D ya
        img = torch.tensor(original_image)
    if torch.is_tensor(mask):
        msk = mask.clone().detach().cpu()
        if msk.dim() == 3:
            msk = msk[0]
    else:
        msk = torch.tensor(mask)
    # Asegurar que img y msk son float (img) y binaria (msk)
    img = img.float()
    msk = (msk > 0).float()
    # Aplicar máscara: píxeles no revelados a 0
    revealed_img = img * msk
    # Convertir a escala de 0-255 uint8 para guardar
    # Suponemos original_image normalizada 0-1 (como ToTensor de MNIST)
    revealed_np = (revealed_img.numpy() * 255).astype(np.uint8)
    # Utilizar PIL para guardar en modo escala de grises
    try:
        from PIL import Image
    except ImportError:
        # Si PIL no está disponible, intentar con OpenCV (cv2)
        try:
            import cv2
            cv2.imwrite(filepath, revealed_np)
            return
        except ImportError:
            raise RuntimeError("No se encontró PIL ni cv2 para guardar imágenes")
    img_pil = Image.fromarray(revealed_np, mode='L')
    img_pil.save(filepath)

def save_play(metadata, filepath):
    """
    Guarda los metadatos de un debate (por ejemplo, secuencia de revelaciones, clases, logits, etc.) en un archivo JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def save_colored_debate(original_image, debate_moves, filepath):
    """
    Guarda una imagen PNG que muestra los píxeles del debate coloreados por agente y numerados por orden.
    
    Args:
        original_image: imagen original (Tensor HxW o 1xHxW)
        debate_moves: lista de movimientos [(y, x, agent_type, move_number), ...]
                     donde agent_type es 'honest' o 'liar'
        filepath: ruta donde guardar la imagen
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
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
        new_H, new_W = H * scale_factor, W * scale_factor
        
        # Convertir a RGB y escalar
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
        img_pil = Image.fromarray(img_rgb, mode='RGB')
        img_pil = img_pil.resize((new_W, new_H), Image.NEAREST)  # NEAREST para mantener píxeles nítidos
        
        draw = ImageDraw.Draw(img_pil)
        
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
        
        img_pil.save(filepath)
        
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
    Registra los resultados de un experimento en un archivo CSV con formato unificado.
    `results` debe ser un diccionario que contenga las claves esperadas.
    """
    # Determinar el tipo de log basado en el nombre del archivo
    if "judges.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "epochs","batch_size","lr","best_loss", "pixels", "accuracy","note"]
    elif "evaluations.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "n_images", "pixels", "accuracy","note"]
    elif "debates.csv" in logfile:
        columns = ["timestamp","judge_name","resolution","thr","seed", "rollouts","n_images","agent_type", "pixels", "started","precommit", "accuracy","note"]
    else:
        # Default columns for backward compatibility
        columns = ["timestamp","judge_name","resolution","thr","seed", "pixels", "accuracy","note"]

    # Si el archivo no existe o está vacío, escribir encabezado
    file_exists = os.path.isfile(logfile)
    write_header = not file_exists or os.path.getsize(logfile) == 0

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if write_header:
            writer.writeheader()
        # Asegurar que results contiene todas las columnas (rellenar si falta alguna no esencial)
        for col in columns:
            if col not in results:
                results[col] = ""
        writer.writerow(results)
