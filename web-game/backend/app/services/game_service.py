"""
Game service that manages the core game logic and state.
Orchestrates between AI agents, judge models, and game state.
"""

import uuid
import time
from typing import Dict, Optional, Tuple, List
import torch

from app.models.game import (
    GameConfig, GameState, GameMove, PixelCoordinate, 
    JudgePrediction, GameStatus, PlayerType, PlayerRole,
    AgentType, GameResult
)
from app.services.ai_interface import GameEnvironment

class GameService:
    """Main service for managing game logic and state"""
    
    def __init__(self):
        self.active_games: Dict[str, GameState] = {}
        self.game_environments: Dict[str, GameEnvironment] = {}
    
    def create_game(self, config: GameConfig) -> Tuple[str, GameState, List[List[float]]]:
        """
        Create a new game with the given configuration
        
        Returns:
            Tuple of (game_id, initial_game_state, image_data)
        """
        game_id = str(uuid.uuid4())
        
        # Initialize game environment
        env = GameEnvironment(config.judge_name)
        agent_config = {}
        if config.agent_type == AgentType.MCTS:
            agent_config['rollouts'] = 100  # Default for web game
        
        # Load image and initialize AI agent
        image, true_label = env.initialize_game(
            agent_type=config.agent_type,
            player_role=config.player_role,
            precommit=config.precommit,
            seed=config.seed,
            agent_config=agent_config,
            threshold=config.threshold
        )
        
        # Create initial game state
        current_time = time.time()
        initial_mask = [[0 for _ in range(28)] for _ in range(28)]
        
        game_state = GameState(
            game_id=game_id,
            status=GameStatus.IN_PROGRESS,
            config=config,
            current_turn=0,
            moves=[],
            current_mask=initial_mask,
            judge_prediction=None,
            true_label=true_label,
            winner=None,
            created_at=current_time,
            completed_at=None
        )
        
        # Store game state and environment
        self.active_games[game_id] = game_state
        self.game_environments[game_id] = env
        
        # Get initial image data for frontend display
        image_np = image.cpu().numpy()
        
        # Debug: Log image statistics
        print(f"Backend image stats: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")
        print(f"Sample corner values: top-left={image_np[0,0]:.4f}, center={image_np[14,14]:.4f}, top-right={image_np[0,27]:.4f}")
        
        image_data = image_np.tolist()
        
        return game_id, game_state, image_data
    
    def make_move(self, game_id: str, pixel: PixelCoordinate) -> Tuple[GameState, Optional[GameMove], bool]:
        """
        Make a human player move
        
        Returns:
            Tuple of (updated_game_state, ai_move_if_any, game_over)
        """
        if game_id not in self.active_games:
            raise ValueError(f"Game {game_id} not found")
        
        game_state = self.active_games[game_id]
        env = self.game_environments[game_id]
        
        if game_state.status != GameStatus.IN_PROGRESS:
            raise ValueError(f"Game {game_id} is not in progress")
        
        # Validate move
        if game_state.current_mask[pixel.y][pixel.x] == 1:
            raise ValueError(f"Pixel ({pixel.x}, {pixel.y}) is already revealed")
        
        current_time = time.time()
        
        # Make human move
        success = env.make_human_move(pixel.x, pixel.y)
        if not success:
            raise ValueError("Invalid move")
        
        # Record human move
        human_move = GameMove(
            turn=game_state.current_turn,
            player_type=PlayerType.HUMAN,
            pixel=pixel,
            timestamp=current_time
        )
        game_state.moves.append(human_move)
        game_state.current_turn += 1
        
        # Update mask
        game_state.current_mask[pixel.y][pixel.x] = 1
        
        # Get judge prediction after human move
        predicted_class, probabilities = env.get_judge_prediction()
        game_state.judge_prediction = JudgePrediction(
            predicted_class=predicted_class,
            probabilities=probabilities.tolist(),
            confidence=float(probabilities.max())
        )
        
        ai_move = None
        game_over = False
        
        # Check if game should continue (less than k total moves)
        if game_state.current_turn < game_state.config.k:
            # AI makes next move
            try:
                ai_x, ai_y = env.make_ai_move(game_state.current_turn)
                
                ai_move = GameMove(
                    turn=game_state.current_turn,
                    player_type=PlayerType.AI,
                    pixel=PixelCoordinate(x=ai_x, y=ai_y),
                    timestamp=time.time()
                )
                game_state.moves.append(ai_move)
                game_state.current_turn += 1
                
                # Update mask with AI move
                game_state.current_mask[ai_y][ai_x] = 1
                
                # Get updated judge prediction after AI move
                predicted_class, probabilities = env.get_judge_prediction()
                game_state.judge_prediction = JudgePrediction(
                    predicted_class=predicted_class,
                    probabilities=probabilities.tolist(),
                    confidence=float(probabilities.max())
                )
                
            except Exception as e:
                print(f"Error in AI move: {e}")
                # End game due to AI error
                game_over = True
        
        # Check if game is over
        if game_state.current_turn >= game_state.config.k:
            game_over = True
        
        if game_over:
            self._finalize_game(game_state)
        
        return game_state, ai_move, game_over
    
    def get_game_state(self, game_id: str) -> Tuple[GameState, List[List[float]]]:
        """Get current game state and image data"""
        if game_id not in self.active_games:
            raise ValueError(f"Game {game_id} not found")
        
        game_state = self.active_games[game_id]
        env = self.game_environments[game_id]
        
        # Get current image with revealed pixels
        image_data = env.get_current_image_with_mask()
        
        return game_state, image_data
    
    def _finalize_game(self, game_state: GameState):
        """Finalize a completed game"""
        game_state.status = GameStatus.COMPLETED
        game_state.completed_at = time.time()
        
        # Determine winner based on final judge prediction
        if game_state.judge_prediction is None:
            # This shouldn't happen, but handle gracefully
            game_state.winner = PlayerType.AI
            return
        
        correct_prediction = (game_state.judge_prediction.predicted_class == game_state.true_label)
        
        # Winner depends on player role
        if game_state.config.player_role == PlayerRole.HONEST:
            # Human is honest, so human wins if judge is correct
            game_state.winner = PlayerType.HUMAN if correct_prediction else PlayerType.AI
        else:
            # Human is liar, so human wins if judge is wrong
            game_state.winner = PlayerType.HUMAN if not correct_prediction else PlayerType.AI
    
    def get_game_result(self, game_id: str) -> GameResult:
        """Get final game result with analysis"""
        if game_id not in self.active_games:
            raise ValueError(f"Game {game_id} not found")
        
        game_state = self.active_games[game_id]
        
        if game_state.status != GameStatus.COMPLETED:
            raise ValueError(f"Game {game_id} is not completed yet")
        
        # Calculate game metrics
        human_won = (game_state.winner == PlayerType.HUMAN)
        final_accuracy = 1.0 if (game_state.judge_prediction.predicted_class == game_state.true_label) else 0.0
        total_turns = len(game_state.moves)
        game_duration = game_state.completed_at - game_state.created_at
        
        # Analyze strategies (basic implementation)
        human_moves = [move for move in game_state.moves if move.player_type == PlayerType.HUMAN]
        ai_moves = [move for move in game_state.moves if move.player_type == PlayerType.AI]
        
        human_strategy_summary = {
            "total_moves": len(human_moves),
            "avg_x_position": sum(move.pixel.x for move in human_moves) / len(human_moves) if human_moves else 0,
            "avg_y_position": sum(move.pixel.y for move in human_moves) / len(human_moves) if human_moves else 0,
            "move_times": [move.timestamp for move in human_moves]
        }
        
        ai_strategy_summary = {
            "total_moves": len(ai_moves),
            "avg_x_position": sum(move.pixel.x for move in ai_moves) / len(ai_moves) if ai_moves else 0,
            "avg_y_position": sum(move.pixel.y for move in ai_moves) / len(ai_moves) if ai_moves else 0,
            "agent_type": game_state.config.agent_type.value
        }
        
        result = GameResult(
            game_id=game_id,
            winner=game_state.winner,
            human_won=human_won,
            final_accuracy=final_accuracy,
            total_turns=total_turns,
            game_duration=game_duration,
            human_strategy_summary=human_strategy_summary,
            ai_strategy_summary=ai_strategy_summary
        )
        
        return result
    
    def cleanup_game(self, game_id: str):
        """Clean up game resources"""
        if game_id in self.active_games:
            del self.active_games[game_id]
        if game_id in self.game_environments:
            del self.game_environments[game_id]
    
    def get_active_games_count(self) -> int:
        """Get number of currently active games"""
        return len(self.active_games)
    
    def cleanup_old_games(self, max_age_seconds: int = 3600):
        """Clean up games older than max_age_seconds"""
        current_time = time.time()
        old_games = []
        
        for game_id, game_state in self.active_games.items():
            age = current_time - game_state.created_at
            if age > max_age_seconds:
                old_games.append(game_id)
        
        for game_id in old_games:
            self.cleanup_game(game_id)
        
        return len(old_games)

# Global game service instance
game_service = GameService()