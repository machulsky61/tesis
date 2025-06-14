import torch

class DebateAgent:
    """
    Base class for debate agents. Stores common information like judge model, target classes and original image.
    """
    def __init__(self, judge_model, my_class, opponent_class, precommit, original_image, thr=0.0, allow_all_pixels=False):
        """
        judge_model: judge model (classifier) to evaluate states.
        my_class: class (digit) that this agent defends the image to be.
        opponent_class: class that the opponent agent defends.
        original_image: complete original image (tensor 1xHxW or 2D HxW) with intensities.
        thr: relevance threshold for pixels to consider.
        allow_all_pixels: if True, allows selecting any pixel (not just >thr).
        """
        self.judge = judge_model
        self.my_class = my_class
        self.opp_class = opponent_class
        self.precommit = precommit
        self.allow_all_pixels = allow_all_pixels
        # Ensure original image is 2D HxW for easier calculations
        self.image = original_image.squeeze()
        # Move image to same device as judge model
        self.image = self.image.to(next(judge_model.parameters()).device)
        # Save threshold and precompute relevant pixels (coordinates) according to thr
        self.thr = thr
        
        H, W = self.image.shape[-2], self.image.shape[-1]
        
        if allow_all_pixels:
            # Allow any pixel in the image
            coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
        else:
            # Only relevant pixels (>thr) - original behavior
            mask_relevant = (self.image > self.thr)
            coords = mask_relevant.nonzero(as_tuple=False)
            if coords.numel() == 0:
                # if no pixel exceeds thr, consider all pixels
                coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
        
        # Convert coords tensor to list of int tuples (on CPU for convenience)
        coords = coords.cpu().tolist()
        self.relevant_coords = [(int(y), int(x)) for (y, x) in coords]
    def choose_pixel(self, mask, reveal_count=None):
        """Abstract method that the agent must implement to choose the next pixel to reveal."""
        raise NotImplementedError("This method must be implemented by DebateAgent subclasses")
