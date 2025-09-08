"""
Health check endpoints for monitoring and deployment
"""

from fastapi import APIRouter
from pydantic import BaseModel
import time
import sys
from pathlib import Path

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str
    python_version: str
    active_games: int

class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str
    python_version: str
    active_games: int
    ai_integration: dict
    judge_model: dict
    system_info: dict

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    from app.services.game_service import game_service
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        environment="development",  # TODO: Get from env var
        python_version=sys.version,
        active_games=game_service.get_active_games_count()
    )

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with system information"""
    from app.services.game_service import game_service
    from app.services.ai_interface import GameEnvironment
    
    # Check AI integration
    ai_integration_status = {}
    try:
        # Try to import main project modules
        main_project_path = Path(__file__).parent.parent.parent.parent.parent / "debate_mnist"
        sys.path.insert(0, str(main_project_path))
        
        from agents.greedy_agent import GreedyAgent
        ai_integration_status["greedy_agent"] = "available"
    except ImportError:
        ai_integration_status["greedy_agent"] = "fallback_mode"
    
    try:
        from agents.mcts_fast import FastMCTSAgent
        ai_integration_status["mcts_agent"] = "available"
    except ImportError:
        ai_integration_status["mcts_agent"] = "fallback_mode"
    
    # Check judge model
    judge_status = {}
    try:
        env = GameEnvironment()
        judge_status["model_loaded"] = env.judge.model is not None
        judge_status["model_name"] = env.judge.judge_name
        judge_status["device"] = str(env.judge.device)
    except Exception as e:
        judge_status["error"] = str(e)
        judge_status["model_loaded"] = False
    
    # System info
    system_info = {
        "python_executable": sys.executable,
        "python_path": sys.path[:3],  # First few paths
    }
    
    return DetailedHealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0", 
        environment="development",
        python_version=sys.version,
        active_games=game_service.get_active_games_count(),
        ai_integration=ai_integration_status,
        judge_model=judge_status,
        system_info=system_info
    )