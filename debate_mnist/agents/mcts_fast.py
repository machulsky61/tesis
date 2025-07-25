import torch
import random
from agents.base_agent import DebateAgent

class FastMCTSAgent(DebateAgent):
    """
    Optimized MCTS agent that uses massive batch simulations for GPU.
    Does not maintain explicit tree, but efficiently simulates many possible sequences.
    """
    def __init__(self, judge_model, my_class, opponent_class, precommit, original_image, thr=0.1, 
                 rollouts=512, total_moves=6, is_truth_agent=True, allow_all_pixels=False):
        super().__init__(judge_model, my_class, opponent_class, precommit, original_image, thr, allow_all_pixels)
        self.rollouts = rollouts
        self.total_moves = total_moves
        self.is_truth_agent = is_truth_agent
        self.device = next(judge_model.parameters()).device
        
        # Optimization: precompute coordinates as tensor
        self.relevant_coords_tensor = torch.tensor(
            self.relevant_coords, device=self.device, dtype=torch.long
        )
        
    def _forward_batched(self, inp, batch_size=4096):
        """Forward pass in batches to avoid GPU OOM."""
        outputs = []
        with torch.no_grad():
            for i in range(0, inp.size(0), batch_size):
                batch = inp[i:i+batch_size]
                outputs.append(self.judge(batch))
        return torch.cat(outputs, dim=0)
    
    def _simulate_rollouts(self, base_masks, moves_left):
        """
        Simulates rollouts from each base mask in vectorized manner.
        
        Args:
            base_masks: [n_candidates, H, W] - masks after first move
            moves_left: int - remaining moves in simulation
            
        Returns:
            final_masks: [n_candidates * rollouts, H, W] - final masks of all simulations
        """
        n_candidates = base_masks.size(0)
        H, W = base_masks.shape[1], base_masks.shape[2]
        
        # Expand base_masks for all rollouts
        # Shape: [n_candidates, rollouts, H, W]
        rollout_masks = base_masks.unsqueeze(1).expand(-1, self.rollouts, -1, -1).clone()
        
        if moves_left > 0:
            # For each rollout, generate random sequence of moves
            # Shape: [n_candidates, rollouts, moves_left]
            random_indices = torch.randint(
                0, len(self.relevant_coords), 
                (n_candidates, self.rollouts, moves_left),
                device=self.device
            )
            
            # Convert indices to coordinates
            # Shape: [n_candidates, rollouts, moves_left, 2]
            random_coords = self.relevant_coords_tensor[random_indices]
            
            # Apply moves sequentially (vectorized)
            for move_idx in range(moves_left):
                coords = random_coords[:, :, move_idx]  # [n_candidates, rollouts, 2]
                y_coords = coords[:, :, 0]  # [n_candidates, rollouts]
                x_coords = coords[:, :, 1]  # [n_candidates, rollouts]
                
                # Apply mask: set 1 at selected coordinates
                # Need flatten for advanced indexing
                flat_masks = rollout_masks.view(-1, H, W)
                flat_y = y_coords.view(-1)
                flat_x = x_coords.view(-1)
                batch_indices = torch.arange(flat_masks.size(0), device=self.device)
                
                flat_masks[batch_indices, flat_y, flat_x] = 1.0
                rollout_masks = flat_masks.view(n_candidates, self.rollouts, H, W)
        
        # Flatten to pass to judge: [n_candidates * rollouts, H, W]
        return rollout_masks.view(-1, H, W)
    
    def _evaluate_candidates(self, final_masks):
        """
        Evaluates all final masks using the judge model.
        
        Args:
            final_masks: [n_candidates * rollouts, H, W]
            
        Returns:
            win_rates: [n_candidates] - win rate per candidate
        """
        # Create revealed values
        values_plane = self.image.unsqueeze(0) * final_masks  # Broadcasting
        
        # Create input for judge: [mask, values] as channels
        judge_input = torch.stack([final_masks, values_plane], dim=1)
        
        # Evaluation in batches
        outputs = self._forward_batched(judge_input, batch_size=2048)
        
        # Reshape: [n_candidates, rollouts, 10]
        n_total = outputs.size(0)
        n_candidates = n_total // self.rollouts
        outputs = outputs.view(n_candidates, self.rollouts, -1)
        
        # Calculate win rates according to precommit/no-precommit logic
        if self.precommit:
            # Precommit mode: compare my class vs opponent class
            my_logits = outputs[:, :, self.my_class]        # [n_candidates, rollouts]
            opp_logits = outputs[:, :, self.opp_class]    # [n_candidates, rollouts]
            wins = (my_logits >= opp_logits).float()      # [n_candidates, rollouts]
            win_rates = wins.mean(dim=1)                  # [n_candidates]
            
        else:
            # No-precommit mode
            if self.my_class is not None:
                # Honest agent: maximize probability of my class
                my_logits = outputs[:, :, self.my_class]  # [n_candidates, rollouts]
                predicted_classes = outputs.argmax(dim=2)
                wins = (predicted_classes == self.my_class).float()
                win_rates = wins.mean(dim=1)
            else:
                # Liar agent: minimize probability of opponent's true class
                opp_logits = outputs[:, :, self.opp_class]  # [n_candidates, rollouts]
                # We want opp_class NOT to be the final prediction
                predicted_classes = outputs.argmax(dim=2)   # [n_candidates, rollouts]
                wins = (predicted_classes != self.opp_class).float()
                win_rates = wins.mean(dim=1)
        
        return win_rates
    
    @torch.no_grad()
    def choose_pixel(self, mask, reveal_count=0):
        """
        Selects the best pixel using massive MCTS simulations.
        
        Args:
            mask: [H, W] - current mask of revealed pixels
            reveal_count: int - number of already revealed pixels
            
        Returns:
            (y, x): tuple with coordinates of best pixel, or None if no candidates
        """
        # 1. Get valid candidates (relevant unrevealed pixels)
        candidates = []
        for y, x in self.relevant_coords:
            if mask[y, x].item() == 0:  # Not revealed
                candidates.append((y, x))
        
        if not candidates:
            return None
        
        n_candidates = len(candidates)
        H, W = mask.shape
        
        # 2. Create base masks (state after revealing each candidate)
        base_masks = mask.unsqueeze(0).expand(n_candidates, -1, -1).clone()
        for i, (y, x) in enumerate(candidates):
            base_masks[i, y, x] = 1.0
        
        # 3. Calculate remaining moves
        moves_left = max(0, self.total_moves - (reveal_count + 1))
        
        # 4. Simulate rollouts from each candidate
        final_masks = self._simulate_rollouts(base_masks, moves_left)
        
        # 5. Evaluate all rollouts
        win_rates = self._evaluate_candidates(final_masks)
        
        # 6. Select best candidate
        best_idx = torch.argmax(win_rates).item()
        best_pixel = candidates[best_idx]
        
        return best_pixel


# Alias for compatibility with existing code
MCTSAgent = FastMCTSAgent