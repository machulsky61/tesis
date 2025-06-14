import torch
from agents.base_agent import DebateAgent

class GreedyAgent(DebateAgent):
    """
    Agent that selects its next move greedily.
    Evaluates each possible unrevealed pixel and chooses the one that most benefits its case immediately.
    """
    def __init__(self, judge_model, my_class, opponent_class, precommit, original_image, thr=0.0, allow_all_pixels=False):
        super().__init__(judge_model, my_class, opponent_class, precommit, original_image, thr, allow_all_pixels)
    def choose_pixel(self, mask, reveal_count=None):
        # mask: 2D tensor (H x W) with 1 on already revealed pixels, 0 on unrevealed
        # We ignore reveal_count since this agent doesn't plan ahead, only current step
        # Get list of candidate pixels (unrevealed) that are relevant
        candidates = [(y, x) for (y, x) in self.relevant_coords if mask[y, x].item() == 0]
        if not candidates:
            return None  # no possible moves
        # Prepare a batch of inputs for the judge, one per hypothetically revealed candidate pixel
        inputs = []
        target_class = self.my_class
        other_class = self.opp_class
        # Create base tensors for current mask and current values
        mask_current = mask.clone().detach()
        mask_current = mask_current.to(self.image.device).float()
        values_current = self.image * mask_current
        H, W = mask_current.shape[-2], mask_current.shape[-1]
        # Create batch of inputs
        for (y, x) in candidates:
            # Clone current mask and add candidate pixel
            mask_new = mask_current.clone()
            mask_new[y, x] = 1.0
            # Calculate new revealed values (adding revealed pixel)
            values_new = self.image * mask_new
            # Form 2xHxW input for judge
            inp = torch.stack([mask_new, values_new], dim=0)
            inputs.append(inp)
        # Join all in batch
        batch_input = torch.stack(inputs, dim=0)  # shape [len(candidates), 2, H, W]
        # Pass batch through judge model
        with torch.no_grad():
            outputs = self.judge(batch_input)
        
        if self.precommit:  # If agent is precommitted, return pixel that maximizes difference between target and opponent class
            # Extract logits of target class and opponent class for each candidate
            # outputs shape: [N, 10], where N = number of candidates
            target_logits = outputs[:, target_class]
            other_logits = outputs[:, other_class]
            # Calculate differences (how much it favors target class over opponent)
            diff = target_logits - other_logits
            # Choose index of best pixel (max difference)
            best_idx = torch.argmax(diff).item()
            best_pixel = candidates[best_idx]
            return best_pixel
        else:
            # If not precommit then check if it's the liar or honest agent
            if self.my_class is not None:  # If agent has target class it's honest -> choose pixel that maximizes probability of that class over all others
                target_logits = outputs[:, self.my_class]
                # Choose index of best pixel (highest probability of target class)
                best_idx = torch.argmax(target_logits).item()
                best_pixel = candidates[best_idx]
                return best_pixel
            else:  # If no target class it's the liar -> choose pixel that minimizes probability of opponent class
                other_logits = outputs[:, self.opp_class]
                # Choose index
                best_idx = torch.argmin(other_logits).item()
                best_pixel = candidates[best_idx]
                return best_pixel