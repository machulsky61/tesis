import torch
from agents.base_agent import DebateAgent

class GreedyAdversarialAgent(DebateAgent):
    """
    Adversarial agent that greedily selects pixels to MINIMIZE 
    the logits of the true class. Simulates an optimal liar agent that 
    tries to confuse the judge by choosing the most harmful pixels.
    
    This agent is useful for evaluating judge robustness against adversarial strategies.
    """
    def __init__(self, judge_model, true_class, original_image, thr=0.0, allow_all_pixels=False):
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
    
    def choose_pixel(self, mask, reveal_count=None):
        """
        Selects the pixel that most minimizes the logits of the true class.
        Equivalent to the liar agent strategy in no-precommit mode.
        """
        # Get list of candidate pixels (unrevealed) that are relevant
        candidates = [(y, x) for (y, x) in self.relevant_coords if mask[y, x].item() == 0]
        
        if not candidates:
            return None  # No possible moves
        
        # Prepare a batch of inputs for the judge, one per candidate pixel
        inputs = []
        mask_current = mask.clone().detach()
        mask_current = mask_current.to(self.image.device).float()
        values_current = self.image * mask_current
        H, W = mask_current.shape[-2], mask_current.shape[-1]
        
        # Create batch of inputs
        for (y, x) in candidates:
            # Clone current mask and add candidate pixel
            mask_new = mask_current.clone()
            mask_new[y, x] = 1.0
            # Calculate new revealed values
            values_new = self.image * mask_new
            # Form 2xHxW input for judge
            inp = torch.stack([mask_new, values_new], dim=0)
            inputs.append(inp)
        
        # Join all in batch
        batch_input = torch.stack(inputs, dim=0)  # shape [len(candidates), 2, H, W]
        
        # Pass batch through judge model
        with torch.no_grad():
            outputs = self.judge(batch_input)
        
        # Adversarial strategy: minimize logits of true class
        true_class_logits = outputs[:, self.true_class]
        
        # Choose pixel that minimizes logits of true class
        best_idx = torch.argmin(true_class_logits).item()
        best_pixel = candidates[best_idx]
        
        return best_pixel