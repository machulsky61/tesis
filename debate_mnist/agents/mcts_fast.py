import torch
import random
from agents.base_agent import DebateAgent

class FastMCTSAgent(DebateAgent):
    """
    Agente MCTS optimizado que usa simulaciones masivas en batch para GPU.
    No mantiene árbol explícito, pero simula muchas secuencias posibles de manera eficiente.
    """
    def __init__(self, judge_model, my_class, opponent_class, precommit, original_image, thr=0.1, 
                 rollouts=512, total_moves=6, is_truth_agent=True, allow_all_pixels=False):
        super().__init__(judge_model, my_class, opponent_class, precommit, original_image, thr, allow_all_pixels)
        self.rollouts = rollouts
        self.total_moves = total_moves
        self.is_truth_agent = is_truth_agent
        self.device = next(judge_model.parameters()).device
        
        # Optimización: precomputar coordenadas como tensor
        self.relevant_coords_tensor = torch.tensor(
            self.relevant_coords, device=self.device, dtype=torch.long
        )
        
    def _forward_batched(self, inp, batch_size=4096):
        """Forward pass en batches para evitar OOM en GPU."""
        outputs = []
        with torch.no_grad():
            for i in range(0, inp.size(0), batch_size):
                batch = inp[i:i+batch_size]
                outputs.append(self.judge(batch))
        return torch.cat(outputs, dim=0)
    
    def _simulate_rollouts(self, base_masks, moves_left):
        """
        Simula rollouts desde cada máscara base de manera vectorizada.
        
        Args:
            base_masks: [n_candidates, H, W] - máscaras después del primer movimiento
            moves_left: int - movimientos restantes en la simulación
            
        Returns:
            final_masks: [n_candidates * rollouts, H, W] - máscaras finales de todas las simulaciones
        """
        n_candidates = base_masks.size(0)
        H, W = base_masks.shape[1], base_masks.shape[2]
        
        # Expandir base_masks para todos los rollouts
        # Shape: [n_candidates, rollouts, H, W]
        rollout_masks = base_masks.unsqueeze(1).expand(-1, self.rollouts, -1, -1).clone()
        
        if moves_left > 0:
            # Para cada rollout, generar secuencia aleatoria de movimientos
            # Shape: [n_candidates, rollouts, moves_left]
            random_indices = torch.randint(
                0, len(self.relevant_coords), 
                (n_candidates, self.rollouts, moves_left),
                device=self.device
            )
            
            # Convertir índices a coordenadas
            # Shape: [n_candidates, rollouts, moves_left, 2]
            random_coords = self.relevant_coords_tensor[random_indices]
            
            # Aplicar movimientos secuencialmente (vectorizado)
            for move_idx in range(moves_left):
                coords = random_coords[:, :, move_idx]  # [n_candidates, rollouts, 2]
                y_coords = coords[:, :, 0]  # [n_candidates, rollouts]
                x_coords = coords[:, :, 1]  # [n_candidates, rollouts]
                
                # Aplicar máscara: poner 1 en las coordenadas seleccionadas
                # Necesitamos flatten para indexing avanzado
                flat_masks = rollout_masks.view(-1, H, W)
                flat_y = y_coords.view(-1)
                flat_x = x_coords.view(-1)
                batch_indices = torch.arange(flat_masks.size(0), device=self.device)
                
                flat_masks[batch_indices, flat_y, flat_x] = 1.0
                rollout_masks = flat_masks.view(n_candidates, self.rollouts, H, W)
        
        # Flatten para pasar al juez: [n_candidates * rollouts, H, W]
        return rollout_masks.view(-1, H, W)
    
    def _evaluate_candidates(self, final_masks):
        """
        Evalúa todas las máscaras finales usando el modelo juez.
        
        Args:
            final_masks: [n_candidates * rollouts, H, W]
            
        Returns:
            win_rates: [n_candidates] - tasa de victoria por candidato
        """
        # Crear valores revelados
        values_plane = self.image.unsqueeze(0) * final_masks  # Broadcasting
        
        # Crear input para el juez: [mask, values] como canales
        judge_input = torch.stack([final_masks, values_plane], dim=1)
        
        # Evaluación en batches
        outputs = self._forward_batched(judge_input, batch_size=2048)
        
        # Reshape: [n_candidates, rollouts, 10]
        n_total = outputs.size(0)
        n_candidates = n_total // self.rollouts
        outputs = outputs.view(n_candidates, self.rollouts, -1)
        
        # Calcular win rates según la lógica de precommit/no-precommit
        if self.precommit:
            # Modo precommit: comparar mi clase vs clase oponente
            my_logits = outputs[:, :, self.my_class]        # [n_candidates, rollouts]
            opp_logits = outputs[:, :, self.opp_class]    # [n_candidates, rollouts]
            wins = (my_logits >= opp_logits).float()      # [n_candidates, rollouts]
            win_rates = wins.mean(dim=1)                  # [n_candidates]
            
        else:
            # Modo no-precommit
            if self.my_class is not None:
                # Agente honesto: maximizar probabilidad de mi clase
                my_logits = outputs[:, :, self.my_class]  # [n_candidates, rollouts]
                predicted_classes = outputs.argmax(dim=2)
                wins = (predicted_classes == self.my_class).float()
                win_rates = wins.mean(dim=1)
            else:
                # Agente mentiroso: minimizar probabilidad de la clase verdadera del oponente
                opp_logits = outputs[:, :, self.opp_class]  # [n_candidates, rollouts]
                # Queremos que opp_class NO sea la predicción final
                predicted_classes = outputs.argmax(dim=2)   # [n_candidates, rollouts]
                wins = (predicted_classes != self.opp_class).float()
                win_rates = wins.mean(dim=1)
        
        return win_rates
    
    @torch.no_grad()
    def choose_pixel(self, mask, reveal_count=0):
        """
        Selecciona el mejor píxel usando simulaciones masivas MCTS.
        
        Args:
            mask: [H, W] - máscara actual de píxeles revelados
            reveal_count: int - número de píxeles ya revelados
            
        Returns:
            (y, x): tupla con coordenadas del mejor píxel, o None si no hay candidatos
        """
        # 1. Obtener candidatos válidos (píxeles relevantes no revelados)
        candidates = []
        for y, x in self.relevant_coords:
            if mask[y, x].item() == 0:  # No revelado
                candidates.append((y, x))
        
        if not candidates:
            return None
        
        n_candidates = len(candidates)
        H, W = mask.shape
        
        # 2. Crear máscaras base (estado después de revelar cada candidato)
        base_masks = mask.unsqueeze(0).expand(n_candidates, -1, -1).clone()
        for i, (y, x) in enumerate(candidates):
            base_masks[i, y, x] = 1.0
        
        # 3. Calcular movimientos restantes
        moves_left = max(0, self.total_moves - (reveal_count + 1))
        
        # 4. Simular rollouts desde cada candidato
        final_masks = self._simulate_rollouts(base_masks, moves_left)
        
        # 5. Evaluar todos los rollouts
        win_rates = self._evaluate_candidates(final_masks)
        
        # 6. Seleccionar el mejor candidato
        best_idx = torch.argmax(win_rates).item()
        best_pixel = candidates[best_idx]
        
        return best_pixel


# Alias para compatibilidad con el código existente
MCTSAgent = FastMCTSAgent