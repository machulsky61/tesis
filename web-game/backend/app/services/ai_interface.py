"""
Interface to integrate with existing AI agents from the main debate_mnist project.
This module adapts the existing agents to work with the web game.
"""

import sys
import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from pathlib import Path

# Add the main project to Python path for imports
main_project_path = Path(__file__).parent.parent.parent.parent.parent / "debate_mnist"
sys.path.insert(0, str(main_project_path))

try:
    from agents.greedy_agent import GreedyAgent
    from agents.mcts_fast import FastMCTSAgent  
    from models.sparse_cnn import SparseCNN
    from utils.helpers import set_seed
except ImportError as e:
    print(f"Warning: Could not import from main project: {e}")
    print("Running in standalone mode - AI agents will use dummy implementations")

from app.models.game import AgentType, PlayerRole, PlayerType

class AIAgentInterface:
    """Interface to wrap existing AI agents for web game use"""
    
    def __init__(self, agent_type: AgentType, judge_model, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.judge_model = judge_model
        self.config = config
        self.agent = None
        
    def initialize_agent(self, role: PlayerRole, image: torch.Tensor, true_label: int, precommit: bool = False):
        """Initialize the AI agent for a new game"""
        try:
            # Determine classes based on role
            if role == PlayerRole.HONEST:
                my_class = true_label
                opp_class = (true_label + 1) % 10  # Just pick a different class
            else:  # LIAR
                my_class = None if not precommit else (true_label + 1) % 10
                opp_class = true_label
            
            # Create agent based on type
            if self.agent_type == AgentType.GREEDY:
                self.agent = GreedyAgent(
                    judge_model=self.judge_model,
                    my_class=my_class,
                    opponent_class=opp_class,
                    precommit=precommit,
                    original_image=image,
                    thr=0.0,  # Allow all pixels for web game
                    allow_all_pixels=True
                )
            elif self.agent_type == AgentType.MCTS:
                rollouts = self.config.get('rollouts', 100)  # Default rollouts for web game
                self.agent = FastMCTSAgent(
                    judge_model=self.judge_model,
                    my_class=my_class,
                    opponent_class=opp_class,
                    precommit=precommit,
                    original_image=image,
                    rollouts=rollouts,
                    thr=0.0,
                    allow_all_pixels=True
                )
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")
                
        except Exception as e:
            print(f"Error initializing agent: {e}")
            # Fallback to dummy agent
            self.agent = DummyAgent(image.shape)
    
    def choose_pixel(self, current_mask: torch.Tensor, turn: int) -> Tuple[int, int]:
        """Have the AI agent choose the next pixel to reveal"""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        try:
            pixel = self.agent.choose_pixel(current_mask, reveal_count=turn)
            if pixel is None:
                # No valid moves - should not happen in normal gameplay
                raise RuntimeError("AI agent could not select a pixel")
            
            return pixel  # Should be (y, x) tuple
            
        except Exception as e:
            print(f"Error in AI pixel selection: {e}")
            # Fallback to random selection
            return self._random_pixel_fallback(current_mask)
    
    def _random_pixel_fallback(self, current_mask: torch.Tensor) -> Tuple[int, int]:
        """Fallback to random pixel selection if AI fails"""
        unrevealed = (current_mask == 0).nonzero()
        if len(unrevealed) > 0:
            idx = random.randint(0, len(unrevealed) - 1)
            return tuple(unrevealed[idx].tolist())
        else:
            raise RuntimeError("No unrevealed pixels available")

class DummyAgent:
    """Dummy agent for testing when main project is not available"""
    
    def __init__(self, image_shape):
        self.image_shape = image_shape
        
    def choose_pixel(self, mask, reveal_count=None):
        """Select a random unrevealed pixel"""
        unrevealed = (mask == 0).nonzero()
        if len(unrevealed) > 0:
            idx = random.randint(0, len(unrevealed) - 1)
            return tuple(unrevealed[idx].tolist())
        return None

class JudgeModelInterface:
    """Interface to wrap existing judge models for web game use"""
    
    def __init__(self, judge_name: str = "28"):
        self.judge_name = judge_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the judge model from the main project"""
        try:
            model_path = main_project_path / "models" / f"{self.judge_name}.pth"
            if not model_path.exists():
                print(f"Warning: Judge model {model_path} not found, using dummy model")
                self.model = DummyJudge()
                return
            
            # Load the SparseCNN model
            self.model = SparseCNN(resolution=28)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded judge model: {self.judge_name}")
            
        except Exception as e:
            print(f"Error loading judge model: {e}")
            self.model = DummyJudge()
    
    def predict(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[int, np.ndarray]:
        """
        Get judge prediction for current image state
        
        Args:
            image: Original image tensor (28x28)
            mask: Binary mask of revealed pixels (28x28)
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Judge model not loaded")
        
        try:
            with torch.no_grad():
                # Create 2-channel input as expected by SparseCNN
                revealed_values = image * mask
                judge_input = torch.stack([mask, revealed_values], dim=0)  # [2, 28, 28]
                judge_input = judge_input.unsqueeze(0).to(self.device)  # [1, 2, 28, 28]
                
                # Get prediction
                logits = self.model(judge_input)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class = int(probabilities.argmax())
                
                return predicted_class, probabilities
                
        except Exception as e:
            print(f"Error in judge prediction: {e}")
            # Fallback to random prediction
            probabilities = np.random.random(10)
            probabilities = probabilities / probabilities.sum()
            predicted_class = int(probabilities.argmax())
            return predicted_class, probabilities

class DummyJudge:
    """Dummy judge for testing when main project is not available"""
    
    def __call__(self, input_tensor):
        """Return random logits"""
        batch_size = input_tensor.shape[0]
        return torch.randn(batch_size, 10)

class GameEnvironment:
    """Main environment that coordinates AI agents and judge"""
    
    def __init__(self, judge_name: str = "28"):
        self.judge = JudgeModelInterface(judge_name)
        self.current_agent = None
        self.current_image = None
        self.current_mask = None
        
    def load_test_image(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Load a random test image from MNIST"""
        if seed is not None:
            set_seed(seed)
            
        try:
            # Try to load from main project
            from utils.data_utils import load_datasets
            _, test_loader = load_datasets(resolution=28, batch_size=1)
            
            # Get a random image by randomly selecting an index
            dataset_size = len(test_loader.dataset)
            if seed is not None:
                random.seed(seed)
            random_idx = random.randint(0, dataset_size - 1)
            
            # Skip to the random index
            for i, (images, labels) in enumerate(test_loader):
                if i == random_idx:
                    image = images[0, 0]  # Remove batch and channel dims -> [28, 28]
                    label = int(labels[0])
                    return image, label
                
        except Exception as e:
            print(f"Error loading MNIST data: {e}")
            # Fallback to dummy image
            if seed is not None:
                torch.manual_seed(seed)
            image = torch.rand(28, 28)
            label = random.randint(0, 9)
            return image, label
    
    def initialize_game(self, agent_type: AgentType, player_role: PlayerRole, 
                       precommit: bool = False, seed: Optional[int] = None,
                       agent_config: Optional[Dict[str, Any]] = None,
                       threshold: float = 0.001) -> Tuple[torch.Tensor, int]:
        """Initialize a new game with specified configuration"""
        if agent_config is None:
            agent_config = {}
            
        # Store threshold for pixel validation
        self.threshold = threshold
            
        # Load test image
        image, true_label = self.load_test_image(seed)
        self.current_image = image
        self.current_mask = torch.zeros(28, 28)
        
        # Determine AI role (opposite of player)
        ai_role = PlayerRole.LIAR if player_role == PlayerRole.HONEST else PlayerRole.HONEST
        
        # Initialize AI agent
        self.current_agent = AIAgentInterface(agent_type, self.judge.model, agent_config)
        self.current_agent.initialize_agent(ai_role, image, true_label, precommit)
        
        return image, true_label
    
    def make_human_move(self, x: int, y: int) -> bool:
        """Make a move for the human player"""
        if self.current_mask[y, x] == 1:
            return False  # Pixel already revealed
        
        # Check if pixel respects threshold
        pixel_value = self.current_image[y, x].item()
        if pixel_value < self.threshold:
            print(f"Human attempted to select pixel below threshold: {pixel_value} < {self.threshold}")
            return False  # Pixel below threshold
        
        self.current_mask[y, x] = 1
        return True
    
    def _get_available_pixels_mask(self) -> torch.Tensor:
        """Get mask of pixels that are unrevealed and above or equal to threshold"""
        # Pixels must be unrevealed and above or equal to threshold
        unrevealed_mask = (self.current_mask == 0)
        threshold_mask = (self.current_image >= self.threshold)
        return unrevealed_mask & threshold_mask
    
    def make_ai_move(self, turn: int) -> Tuple[int, int]:
        """Make a move for the AI player"""
        # Get available pixels that respect threshold
        available_mask = self._get_available_pixels_mask()
        
        # If no pixels respect threshold, fall back to any unrevealed pixel
        if available_mask.sum() == 0:
            print(f"Warning: No pixels above threshold {self.threshold}, falling back to any unrevealed pixel")
            available_mask = (self.current_mask == 0)
        
        # Temporarily modify mask to restrict AI choices
        original_mask = self.current_mask.clone()
        # Set unavailable pixels as revealed so AI won't choose them
        self.current_mask = torch.where(available_mask, self.current_mask, torch.ones_like(self.current_mask))
        
        try:
            pixel = self.current_agent.choose_pixel(self.current_mask, turn)
            y, x = pixel  # AI returns (y, x)
        except Exception as e:
            print(f"AI agent failed to choose pixel: {e}")
            # Fall back to random selection from available pixels
            available_pixels = available_mask.nonzero()
            if len(available_pixels) == 0:
                # Truly no pixels available, choose any unrevealed
                available_pixels = (original_mask == 0).nonzero()
            if len(available_pixels) > 0:
                idx = random.randint(0, len(available_pixels) - 1)
                y, x = available_pixels[idx].tolist()
            else:
                raise RuntimeError("No pixels available for AI to choose")
        finally:
            # Restore original mask and reveal chosen pixel
            self.current_mask = original_mask
            self.current_mask[y, x] = 1
        
        return x, y  # Return (x, y) for web frontend
    
    def get_judge_prediction(self) -> Tuple[int, np.ndarray]:
        """Get current judge prediction"""
        return self.judge.predict(self.current_image, self.current_mask)
    
    def get_current_image_with_mask(self) -> np.ndarray:
        """Get current image state for display"""
        revealed_image = (self.current_image * self.current_mask).cpu().numpy()
        return revealed_image.tolist()  # Convert to list for JSON serialization