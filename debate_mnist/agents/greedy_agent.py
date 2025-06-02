import torch
from agents.base_agent import DebateAgent

class GreedyAgent(DebateAgent):
    """
    Agente que selecciona su siguiente movimiento de forma codiciosa (greedy).
    Evalúa cada posible píxel no revelado y elige aquel que más beneficia a su caso inmediatamente.
    """
    def choose_pixel(self, mask, reveal_count=None):
        # mask: tensor 2D (H x W) con 1 en píxeles ya revelados, 0 en no revelados
        # Ignoramos reveal_count ya que este agente no planifica a futuro, solo paso actual
        # Obtener lista de píxeles candidatos (no revelados) que sean relevantes
        candidates = [(y, x) for (y, x) in self.relevant_coords if mask[y, x].item() == 0]
        if not candidates:
            return None  # no hay movimientos posibles
        # Preparar un batch de entradas para el juez, una por cada pixel candidato revelado hipotéticamente
        inputs = []
        target_class = self.my_class
        other_class = self.opp_class
        # Crear tensores base para máscara actual y valores actuales
        mask_current = mask.clone().detach()
        mask_current = mask_current.to(self.image.device).float()
        values_current = self.image * mask_current
        H, W = mask_current.shape[-2], mask_current.shape[-1]
        # Crear batch de entradas
        for (y, x) in candidates:
            # Clonar máscara actual y agregar el pixel candidato
            mask_new = mask_current.clone()
            mask_new[y, x] = 1.0
            # Calcular nuevos valores revelados (sumando pixel revelado)
            values_new = self.image * mask_new
            # Formar entrada 2xHxW para el juez
            inp = torch.stack([mask_new, values_new], dim=0)
            inputs.append(inp)
        # Unir todos en batch
        batch_input = torch.stack(inputs, dim=0)  # shape [len(candidates), 2, H, W]
        # Pasar el batch por el modelo juez
        with torch.no_grad():
            outputs = self.judge(batch_input)
        
        if self.precommit:# Si el agente precomprometido, devuelve el pixel que maximiza la diferencia entre la clase objetivo y la opuesta
            # Extraer logits de la clase objetivo y la clase opuesta para cada candidato
            # outputs shape: [N, 10], donde N = número de candidatos
            target_logits = outputs[:, target_class]
            other_logits = outputs[:, other_class]
            # Calcular diferencias (qué tanto favorece a la clase objetivo sobre la opuesta)
            diff = target_logits - other_logits
            # Escoger el índice del mejor pixel (max diferencia)
            best_idx = torch.argmax(diff).item()
            best_pixel = candidates[best_idx]
            return best_pixel
        else:
            # Si no es precomit entonces veo si es el mentiroso o el honesto
            if self.my_class is not None:# Si el agente tiene una clase objetivo es el honesto -> elegir el pixel que maximiza la probabilidad de esa clase por sobre todas las demás
                target_logits = outputs[:, self.my_class]
                # Escoger el índice del mejor pixel (mayor probabilidad de la clase objetivo)
                best_idx = torch.argmax(target_logits).item()
                best_pixel = candidates[best_idx]
                return best_pixel
            else:# Si no tiene clase objetivo es el mentiroso -> elegir el pixel que minimiza la probabilidad de la clase opuesta
                other_logits = outputs[:, self.opp_class]
                # Escoger el índice
                best_idx = torch.argmin(other_logits).item()
                if best_idx < 0 or best_idx >= len(candidates):
                     import pdb; pdb.set_trace()
                best_pixel = candidates[best_idx]
                return best_pixel