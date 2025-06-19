import torch
import numpy as np
from agents.base_agent import DebateAgent

class MCTSAdversarialAgent(DebateAgent):
    """
    MCTS Adversarial agent that mimics mcts_fast liar behavior in solo evaluation. 
    Chooses ALL k pixels sequentially to maximize wrong predictions.
    
    - No opponent: chooses all k pixels by itself
    - Same vectorized rollout efficiency as mcts_fast  
    - Same liar logic: maximize (predicted_class != true_class)
    - Supports allow_all_pixels for unrestricted selection
    """
    
    def __init__(self, judge_model, true_class, original_image, thr=0.0, 
                 allow_all_pixels=False, rollouts=512):
        # Configure as mcts_fast liar: my_class=None, opponent_class=true_class, precommit=False
        super().__init__(
            judge_model=judge_model, 
            my_class=None,  # Defends no class
            opponent_class=true_class,  # Attacks true class
            precommit=False, 
            original_image=original_image, 
            thr=thr, 
            allow_all_pixels=allow_all_pixels
        )
        self.true_class = true_class
        self.rollouts = rollouts
        self.device = next(judge_model.parameters()).device
        
        # Precompute coordinates tensor for efficiency
        if allow_all_pixels:
            H, W = self.image.shape[-2:]
            all_coords = [(i, j) for i in range(H) for j in range(W)]
            self.all_coords_tensor = torch.tensor(all_coords, device=self.device, dtype=torch.long)
        else:
            self.all_coords_tensor = torch.tensor(
                self.relevant_coords, device=self.device, dtype=torch.long
            )
    
    def choose_pixel(self, mask, reveal_count=None):
        """
        Choose best pixel using MCTS rollouts.
        Uses vectorized evaluation for efficiency with high rollout counts.
        """
        # Get available candidates
        if self.allow_all_pixels:
            H, W = self.image.shape[-2:]
            all_candidates = [(i, j) for i in range(H) for j in range(W)]
        else:
            all_candidates = [(y, x) for (y, x) in self.relevant_coords]
        
        # Filter unrevealed pixels
        candidates = [(y, x) for (y, x) in all_candidates if mask[y, x].item() == 0]
        
        if not candidates:
            return None
        
        n_candidates = len(candidates)
        H, W = mask.shape
        
        # Create base masks after revealing each candidate
        base_masks = mask.unsqueeze(0).expand(n_candidates, -1, -1).clone()
        for i, (y, x) in enumerate(candidates):
            base_masks[i, y, x] = 1.0
        
        # Calculate remaining moves (we choose all remaining pixels)
        moves_left = max(0, 6 - (reveal_count + 1)) if reveal_count is not None else 5
        
        # Simulate rollouts from each candidate (vectorized)
        final_masks = self._simulate_rollouts(base_masks, moves_left)
        
        # Evaluate all rollouts using batch forward pass
        win_rates = self._evaluate_candidates(final_masks, n_candidates)
        
        # Select best candidate
        best_idx = torch.argmax(win_rates).item()
        return candidates[best_idx]
    
    def _simulate_rollouts(self, base_masks, moves_left):
        """
        Vectorized rollout simulation.
        Efficiently generates [n_candidates * rollouts] final masks.
        """
        n_candidates = base_masks.size(0)
        H, W = base_masks.shape[1], base_masks.shape[2]
        
        # Expand base_masks for all rollouts: [n_candidates, rollouts, H, W]
        rollout_masks = base_masks.unsqueeze(1).expand(-1, self.rollouts, -1, -1).clone()
        
        if moves_left > 0:
            # Generate random moves for completing the masks
            # Shape: [n_candidates, rollouts, moves_left]
            n_coords = len(self.all_coords_tensor)
            random_indices = torch.randint(
                0, n_coords,
                (n_candidates, self.rollouts, moves_left),
                device=self.device
            )
            
            # Convert to coordinates: [n_candidates, rollouts, moves_left, 2]
            random_coords = self.all_coords_tensor[random_indices]
            
            # Apply moves sequentially (vectorized)
            for move_idx in range(moves_left):
                coords = random_coords[:, :, move_idx]  # [n_candidates, rollouts, 2]
                y_coords = coords[:, :, 0]  # [n_candidates, rollouts]
                x_coords = coords[:, :, 1]  # [n_candidates, rollouts]
                
                # Apply mask using advanced indexing
                flat_masks = rollout_masks.view(-1, H, W)
                flat_y = y_coords.view(-1)
                flat_x = x_coords.view(-1)
                batch_indices = torch.arange(flat_masks.size(0), device=self.device)
                
                flat_masks[batch_indices, flat_y, flat_x] = 1.0
                rollout_masks = flat_masks.view(n_candidates, self.rollouts, H, W)
        
        # Flatten for batch evaluation: [n_candidates * rollouts, H, W]
        return rollout_masks.view(-1, H, W)
    
    def _evaluate_candidates(self, final_masks, n_candidates):
        """
        Batch evaluation but with liar logic.
        """
        # Create revealed values
        values_plane = self.image.unsqueeze(0) * final_masks  # Broadcasting
        
        # Create input for judge: [mask, values] as channels
        judge_input = torch.stack([final_masks, values_plane], dim=1)
        
        # Forward pass in batches to avoid OOM )
        outputs = self._forward_batched(judge_input, batch_size=2048)
        
        # Reshape: [n_candidates, rollouts, 10]
        outputs = outputs.view(n_candidates, self.rollouts, -1)
        
        # Liar logic from mcts_fast line 127: wins = (predicted_classes != self.opp_class).float()
        predicted_classes = outputs.argmax(dim=2)  # [n_candidates, rollouts]
        wins = (predicted_classes != self.true_class).float()  # [n_candidates, rollouts]
        win_rates = wins.mean(dim=1)  # [n_candidates]
        
        return win_rates
    
    def _forward_batched(self, inp, batch_size=2048):
        """Forward pass in batches"""
        outputs = []
        with torch.no_grad():
            for i in range(0, inp.size(0), batch_size):
                batch = inp[i:i+batch_size]
                outputs.append(self.judge(batch))
        return torch.cat(outputs, dim=0)
    
    def get_strategy_name(self):
        """Returns a descriptive name for this agent strategy."""
        return f"MCTS_Adversarial_{self.rollouts}r"
    
    def __str__(self):
        return f"MCTSAdversarialAgent(rollouts={self.rollouts}, target_class={self.true_class})"