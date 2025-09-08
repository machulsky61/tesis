from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

class PlayerRole(str, Enum):
    """Player role in the debate"""
    HONEST = "honest"
    LIAR = "liar"

class AgentType(str, Enum):
    """AI agent type"""
    GREEDY = "greedy"
    MCTS = "mcts"

class GameStatus(str, Enum):
    """Game status"""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class PlayerType(str, Enum):
    """Type of player making a move"""
    HUMAN = "human"
    AI = "ai"

class GameConfig(BaseModel):
    """Configuration for a new game"""
    k: int = Field(default=6, ge=2, le=12, description="Total number of pixels to reveal")
    player_role: PlayerRole = Field(description="Role of human player")
    agent_type: AgentType = Field(default=AgentType.GREEDY, description="Type of AI opponent")
    precommit: bool = Field(default=False, description="Whether to use precommit strategy")
    judge_name: str = Field(default="28", description="Name of judge model to use")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    threshold: float = Field(default=0.001, ge=0.0, le=1.0, description="Minimum pixel intensity for selection (0.0 for OOD mode, 0.001+ for standard)")

class PixelCoordinate(BaseModel):
    """Pixel coordinates"""
    x: int = Field(ge=0, lt=28, description="X coordinate (0-27)")
    y: int = Field(ge=0, lt=28, description="Y coordinate (0-27)")

class GameMove(BaseModel):
    """A move in the debate game"""
    turn: int = Field(ge=0, description="Turn number (0-indexed)")
    player_type: PlayerType = Field(description="Who made the move")
    pixel: PixelCoordinate = Field(description="Selected pixel coordinates")
    timestamp: float = Field(description="Unix timestamp of the move")

class JudgePrediction(BaseModel):
    """Judge model prediction"""
    predicted_class: int = Field(ge=0, le=9, description="Predicted digit class")
    probabilities: List[float] = Field(description="Probability distribution over all classes")
    confidence: float = Field(ge=0.0, le=1.0, description="Max probability")

class GameState(BaseModel):
    """Current state of the game"""
    game_id: str = Field(description="Unique game identifier")
    status: GameStatus = Field(description="Current game status")
    config: GameConfig = Field(description="Game configuration")
    current_turn: int = Field(ge=0, description="Current turn number")
    moves: List[GameMove] = Field(default=[], description="List of moves made")
    current_mask: List[List[int]] = Field(description="Current revealed pixels (28x28 binary mask)")
    judge_prediction: Optional[JudgePrediction] = Field(default=None, description="Latest judge prediction")
    true_label: int = Field(ge=0, le=9, description="True label of the image")
    winner: Optional[PlayerType] = Field(default=None, description="Winner of the game")
    created_at: float = Field(description="Game creation timestamp")
    completed_at: Optional[float] = Field(default=None, description="Game completion timestamp")

class CreateGameRequest(BaseModel):
    """Request to create a new game"""
    config: GameConfig = Field(description="Game configuration")

class CreateGameResponse(BaseModel):
    """Response for game creation"""
    game_id: str = Field(description="Created game ID")
    initial_state: GameState = Field(description="Initial game state")
    image_data: List[List[float]] = Field(description="Base64 encoded image data for display")

class MakeMoveRequest(BaseModel):
    """Request to make a move"""
    pixel: PixelCoordinate = Field(description="Selected pixel coordinates")

class MakeMoveResponse(BaseModel):
    """Response for making a move"""
    success: bool = Field(description="Whether the move was successful")
    game_state: GameState = Field(description="Updated game state")
    ai_move: Optional[GameMove] = Field(default=None, description="AI move if it's AI's turn next")
    game_over: bool = Field(description="Whether the game is over")
    error: Optional[str] = Field(default=None, description="Error message if move failed")

class GetGameStateResponse(BaseModel):
    """Response for getting game state"""
    game_state: GameState = Field(description="Current game state")
    image_data: List[List[float]] = Field(description="Current image with revealed pixels")

class GameResult(BaseModel):
    """Final game result"""
    game_id: str = Field(description="Game identifier")
    winner: PlayerType = Field(description="Winner of the game")
    human_won: bool = Field(description="Whether human player won")
    final_accuracy: float = Field(description="Final judge accuracy (1.0 if correct, 0.0 if wrong)")
    total_turns: int = Field(description="Total number of turns played")
    game_duration: float = Field(description="Game duration in seconds")
    human_strategy_summary: Dict[str, Any] = Field(description="Summary of human strategy")
    ai_strategy_summary: Dict[str, Any] = Field(description="Summary of AI strategy")

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    game_id: Optional[str] = Field(default=None, description="Game ID if relevant")

# Additional models for future features
class GameStats(BaseModel):
    """Game statistics"""
    total_games: int = Field(description="Total games played")
    human_wins: int = Field(description="Number of human wins")
    ai_wins: int = Field(description="Number of AI wins")
    average_game_duration: float = Field(description="Average game duration in seconds")
    win_rate: float = Field(description="Human win rate")

class JudgeInfo(BaseModel):
    """Information about a judge model"""
    name: str = Field(description="Judge model name")
    accuracy: float = Field(description="Model accuracy on test set")
    description: str = Field(description="Model description")
    training_date: str = Field(description="When the model was trained")
    resolution: int = Field(description="Image resolution the model expects")