"""
Game API endpoints for the human vs AI debate game
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models.game import (
    CreateGameRequest, CreateGameResponse, MakeMoveRequest, MakeMoveResponse,
    GetGameStateResponse, GameResult, ErrorResponse, JudgeInfo, GameStats
)
from app.services.game_service import game_service

router = APIRouter()

@router.post("/create", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new debate game"""
    try:
        game_id, game_state, image_data = game_service.create_game(request.config)
        
        return CreateGameResponse(
            game_id=game_id,
            initial_state=game_state,
            image_data=image_data
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create game: {str(e)}"
        )

@router.post("/{game_id}/move", response_model=MakeMoveResponse)
async def make_move(game_id: str, request: MakeMoveRequest):
    """Make a move in the game (human player)"""
    try:
        game_state, ai_move, game_over = game_service.make_move(game_id, request.pixel)
        
        return MakeMoveResponse(
            success=True,
            game_state=game_state,
            ai_move=ai_move,
            game_over=game_over,
            error=None
        )
        
    except ValueError as e:
        return MakeMoveResponse(
            success=False,
            game_state=None,
            ai_move=None,
            game_over=False,
            error=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to make move: {str(e)}"
        )

@router.get("/{game_id}/state", response_model=GetGameStateResponse)
async def get_game_state(game_id: str):
    """Get current game state"""
    try:
        game_state, image_data = game_service.get_game_state(game_id)
        
        return GetGameStateResponse(
            game_state=game_state,
            image_data=image_data
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get game state: {str(e)}"
        )

@router.get("/{game_id}/result", response_model=GameResult)
async def get_game_result(game_id: str):
    """Get final game result with analysis"""
    try:
        result = game_service.get_game_result(game_id)
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get game result: {str(e)}"
        )

@router.delete("/{game_id}")
async def delete_game(game_id: str):
    """Delete/cleanup a game"""
    try:
        game_service.cleanup_game(game_id)
        return {"success": True, "message": f"Game {game_id} deleted"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete game: {str(e)}"
        )

@router.get("/judges", response_model=List[JudgeInfo])
async def get_available_judges():
    """Get list of available judge models"""
    # For MVP, we'll return a static list
    # In future versions, this could scan the models directory
    judges = [
        JudgeInfo(
            name="28",
            accuracy=0.89,
            description="Standard SparseCNN trained on 28x28 resolution",
            training_date="2024-06-01",
            resolution=28
        ),
        JudgeInfo(
            name="16", 
            accuracy=0.85,
            description="SparseCNN trained on 16x16 resolution",
            training_date="2024-06-01",
            resolution=16
        )
    ]
    
    return judges

@router.get("/stats", response_model=GameStats) 
async def get_game_stats():
    """Get overall game statistics"""
    # For MVP, return basic stats
    # In future versions, this would query a proper database
    
    return GameStats(
        total_games=0,  # TODO: Track in database
        human_wins=0,
        ai_wins=0, 
        average_game_duration=0.0,
        win_rate=0.0
    )

@router.post("/cleanup")
async def cleanup_old_games(max_age_hours: int = 1):
    """Clean up old/abandoned games"""
    try:
        max_age_seconds = max_age_hours * 3600
        cleaned_count = game_service.cleanup_old_games(max_age_seconds)
        
        return {
            "success": True,
            "cleaned_games": cleaned_count,
            "message": f"Cleaned up {cleaned_count} old games"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup games: {str(e)}"
        )