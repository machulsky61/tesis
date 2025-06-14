import torch
import numpy as np
from agents.base_agent import DebateAgent
from agents.mcts_fast import FastMCTSAgent

class MCTSAdversarialAgent(DebateAgent):
    """
    Adversarial agent that uses MCTS to strategically select pixels to MINIMIZE 
    the logits of the true class. This is a sophisticated adversarial agent that 
    uses Monte Carlo Tree Search to find the most damaging pixel sequences.
    
    This agent simulates an intelligent adversary that can think ahead multiple moves
    to find the optimal strategy to confuse the judge.
    """
    
    def __init__(self, judge_model, true_class, original_image, thr=0.0, 
                 allow_all_pixels=False, rollouts=100):
        # For this adversarial agent:
        # - my_class = None (defends no class)
        # - opponent_class = true_class (wants to minimize this class)
        # - precommit = False (liar-type no-precommit behavior)
        super().__init__(
            judge_model=judge_model, 
            my_class=None,  # Defends no class
            opponent_class=true_class,  # Wants to minimize true class
            precommit=False, 
            original_image=original_image, 
            thr=thr, 
            allow_all_pixels=allow_all_pixels
        )
        self.true_class = true_class
        self.rollouts = rollouts
        
        # Create an internal MCTS agent but modify its evaluation logic
        self.mcts_engine = FastMCTSAgent(
            judge_model=judge_model,
            my_class=None,  # No class to defend
            opponent_class=true_class,  # Class to attack
            precommit=False,
            original_image=original_image,
            thr=thr,
            rollouts=rollouts,
            allow_all_pixels=allow_all_pixels
        )
    
    def choose_pixel(self, mask, reveal_count=None):
        """
        Uses MCTS to select the pixel that most effectively minimizes 
        the true class logits through strategic lookahead.
        """
        # Get available pixel coordinates
        if self.allow_all_pixels:
            # Can select any pixel
            H, W = self.original_image.shape[-2:]
            all_coords = [(i, j) for i in range(H) for j in range(W)]
        else:
            # Only select pixels above threshold
            image_2d = self.original_image.squeeze()
            coords_above_thr = torch.nonzero(image_2d > self.thr, as_tuple=False).cpu().numpy()
            all_coords = [(int(coord[0]), int(coord[1])) for coord in coords_above_thr]
        
        # Filter out already revealed pixels
        mask_2d = mask.squeeze()
        available_coords = [
            (i, j) for i, j in all_coords 
            if mask_2d[i, j] == 0  # Not yet revealed
        ]
        
        if not available_coords:
            # No pixels available, return random coordinate
            H, W = self.original_image.shape[-2:]
            return (np.random.randint(0, H), np.random.randint(0, W))
        
        # Use MCTS to evaluate each potential move
        best_coord = None
        best_score = float('inf')  # We want to minimize true class logits
        
        device = next(self.judge_model.parameters()).device
        
        for coord in available_coords:
            # Create hypothetical mask with this pixel revealed
            test_mask = mask.clone()
            test_mask[0, 0, coord[0], coord[1]] = 1.0
            
            # Evaluate this move using MCTS lookahead
            score = self._evaluate_adversarial_move(test_mask, coord)
            
            if score < best_score:  # Lower score is better for adversarial agent
                best_score = score
                best_coord = coord
        
        return best_coord if best_coord else available_coords[0]
    
    def _evaluate_adversarial_move(self, mask_with_move, move_coord):
        """
        Evaluates how good a move is for the adversarial agent.
        Uses simplified MCTS rollouts focused on minimizing true class confidence.
        """
        device = next(self.judge_model.parameters()).device
        
        # Immediate evaluation: how much does this single move hurt the true class?
        image_input = torch.stack([mask_with_move.squeeze(), self.original_image.squeeze()], dim=0)
        image_input = image_input.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self.judge_model(image_input)
            true_class_logit = logits[0, self.true_class].item()
        
        # For adversarial agent, lower true class logit is better
        immediate_score = true_class_logit
        
        # Add some rollout simulation for strategic depth
        rollout_penalty = self._simulate_opponent_response(mask_with_move)
        
        # Combine immediate effect with strategic considerations
        total_score = immediate_score + 0.3 * rollout_penalty
        
        return total_score
    
    def _simulate_opponent_response(self, current_mask):
        """
        Simulates how an honest agent might respond to our adversarial move.
        Returns a penalty score based on how easily the honest agent could recover.
        """
        device = next(self.judge_model.parameters()).device
        
        # Count revealed pixels
        revealed_count = (current_mask > 0).sum().item()
        total_pixels = current_mask.numel()
        
        if revealed_count >= total_pixels * 0.8:  # If most pixels revealed
            return 0  # No penalty, game almost over
        
        # Simple simulation: assume honest agent picks best remaining pixel
        image_2d = self.original_image.squeeze()
        mask_2d = current_mask.squeeze()
        
        # Find unrevealed pixels above threshold
        if self.allow_all_pixels:
            H, W = image_2d.shape
            available_coords = [(i, j) for i in range(H) for j in range(W)]
        else:
            coords_above_thr = torch.nonzero(image_2d > self.thr, as_tuple=False).cpu().numpy()
            available_coords = [(int(coord[0]), int(coord[1])) for coord in coords_above_thr]
        
        available_coords = [
            (i, j) for i, j in available_coords 
            if mask_2d[i, j] == 0  # Not yet revealed
        ]
        
        if not available_coords:
            return 0
        
        # Evaluate best honest response
        best_honest_improvement = 0
        
        for coord in available_coords[:min(5, len(available_coords))]:  # Check top 5 to save time
            test_mask = current_mask.clone()
            test_mask[0, 0, coord[0], coord[1]] = 1.0
            
            image_input = torch.stack([test_mask.squeeze(), image_2d], dim=0)
            image_input = image_input.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = self.judge_model(image_input)
                true_class_logit = logits[0, self.true_class].item()
            
            # Current true class logit
            current_input = torch.stack([current_mask.squeeze(), image_2d], dim=0)
            current_input = current_input.unsqueeze(0).to(device)
            
            with torch.no_grad():
                current_logits = self.judge_model(current_input)
                current_true_logit = current_logits[0, self.true_class].item()
            
            improvement = true_class_logit - current_true_logit
            best_honest_improvement = max(best_honest_improvement, improvement)
        
        # Return penalty: higher if honest agent can easily recover
        return best_honest_improvement
    
    def get_strategy_name(self):
        """Returns a descriptive name for this agent strategy."""
        return f"MCTS_Adversarial_{self.rollouts}r"
    
    def __str__(self):
        return f"MCTSAdversarialAgent(rollouts={self.rollouts}, target_class={self.true_class})"