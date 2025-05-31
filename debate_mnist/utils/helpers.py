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

def log_results_csv(logfile, results):
    """
    Registra los resultados de un experimento en un archivo CSV con formato unificado.
    `results` debe ser un diccionario que contenga las claves esperadas.
    """
    # Determinar el tipo de log basado en el nombre del archivo
    if "judges.csv" in logfile:
        columns = ["timestamp", "judge_name", "seed", "resolution", "thr", "k", 
                  "epochs", "batch_size", "lr", "best_loss", "accuracy", "note"]
    elif "evaluations.csv" in logfile:
        columns = ["timestamp", "judge_name", "seed", "resolution", "thr", "k", 
                  "n_images", "accuracy"]
    elif "debates.csv" in logfile:
        columns = ["timestamp", "judge_name", "seed", "resolution", "thr", "k", 
                  "agent_type", "rollouts", "n_images", "accuracy", "precommit", "note", "started"]
    else:
        # Default columns for backward compatibility
        columns = ["timestamp", "seed", "resolution", "thr", "masking_type", 
                  "k", "agent_type", "rollouts", "accuracy"]

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
