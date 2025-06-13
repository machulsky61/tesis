import torch
from agents.base_agent import DebateAgent

class GreedyAdversarialAgent(DebateAgent):
    """
    Agente adversarial que selecciona píxeles de forma codiciosa para MINIMIZAR 
    los logits de la clase verdadera. Simula un agente mentiroso óptimo que 
    intenta confundir al juez eligiendo los píxeles más perjudiciales.
    
    Este agente es útil para evaluar la robustez del juez contra estrategias adversariales.
    """
    def __init__(self, judge_model, true_class, original_image, thr=0.0, allow_all_pixels=False):
        # Para este agente adversarial:
        # - my_class = None (no defiende ninguna clase)
        # - opponent_class = true_class (quiere minimizar esta clase)
        # - precommit = False (comportamiento tipo mentiroso no-precommit)
        super().__init__(
            judge_model=judge_model, 
            my_class=None,  # No defiende ninguna clase
            opponent_class=true_class,  # Quiere minimizar la clase verdadera
            precommit=False, 
            original_image=original_image, 
            thr=thr, 
            allow_all_pixels=allow_all_pixels
        )
        self.true_class = true_class
    
    def choose_pixel(self, mask, reveal_count=None):
        """
        Selecciona el píxel que más minimiza los logits de la clase verdadera.
        Equivale a la estrategia del agente mentiroso en modo no-precommit.
        """
        # Obtener lista de píxeles candidatos (no revelados) que sean relevantes
        candidates = [(y, x) for (y, x) in self.relevant_coords if mask[y, x].item() == 0]
        
        if not candidates:
            return None  # No hay movimientos posibles
        
        # Preparar un batch de entradas para el juez, una por cada pixel candidato
        inputs = []
        mask_current = mask.clone().detach()
        mask_current = mask_current.to(self.image.device).float()
        values_current = self.image * mask_current
        H, W = mask_current.shape[-2], mask_current.shape[-1]
        
        # Crear batch de entradas
        for (y, x) in candidates:
            # Clonar máscara actual y agregar el pixel candidato
            mask_new = mask_current.clone()
            mask_new[y, x] = 1.0
            # Calcular nuevos valores revelados
            values_new = self.image * mask_new
            # Formar entrada 2xHxW para el juez
            inp = torch.stack([mask_new, values_new], dim=0)
            inputs.append(inp)
        
        # Unir todos en batch
        batch_input = torch.stack(inputs, dim=0)  # shape [len(candidates), 2, H, W]
        
        # Pasar el batch por el modelo juez
        with torch.no_grad():
            outputs = self.judge(batch_input)
        
        # Estrategia adversarial: minimizar logits de la clase verdadera
        true_class_logits = outputs[:, self.true_class]
        
        # Escoger el píxel que minimiza los logits de la clase verdadera
        best_idx = torch.argmin(true_class_logits).item()
        best_pixel = candidates[best_idx]
        
        return best_pixel