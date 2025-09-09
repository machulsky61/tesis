import { useState, useEffect, useCallback, useRef } from 'react';
import gameAPI from '../services/api';
import { handleAPIError, isGameOver, DEFAULT_GAME_CONFIG } from '../utils/helpers';

/**
 * Main game hook that manages game state and interactions
 */
export const useGame = () => {
  if (process.env.NODE_ENV === 'development') {
    console.log('useGame hook initializing...');
  }
  
  // Game state
  const [gameState, setGameState] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gameConfig, setGameConfig] = useState(DEFAULT_GAME_CONFIG);
  
  // UI state
  const [selectedPixel, setSelectedPixel] = useState(null);
  const [hoveredPixel, setHoveredPixel] = useState(null);
  const [isPlayerTurn, setIsPlayerTurn] = useState(true);
  const [gameResult, setGameResult] = useState(null);
  
  // Refs for cleanup
  const abortControllerRef = useRef(null);

  /**
   * Fetch game result
   */
  const fetchGameResult = useCallback(async () => {
    if (!gameId) return;

    try {
      const result = await gameAPI.getGameResult(gameId);
      setGameResult(result);
      console.log('Game result:', result);
    } catch (err) {
      console.error('Error fetching game result:', err);
      // Don't set error for result fetch failure - game can continue without it
    }
  }, [gameId]);

  /**
   * Create a new game
   */
  const createGame = useCallback(async (config = gameConfig) => {
    try {
      setLoading(true);
      setError(null);
      setGameResult(null);
      
      // Cancel any ongoing requests
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      // Add random seed if not specified
      const gameConfigWithSeed = {
        ...config,
        seed: config.seed || Math.floor(Math.random() * 1000000),
        threshold: config.allow_zero_pixels ? 0.0 : 0.001 // Standard mode: only pixels > 0
      };
      
      console.log('Creating game with config:', gameConfigWithSeed);
      
      const response = await gameAPI.createGame(gameConfigWithSeed);
      
      console.log('Full API response:', response);
      console.log('Game state structure:', response.initial_state);
      console.log('Current mask structure:', response.initial_state?.current_mask);
      console.log('Mask is array?', Array.isArray(response.initial_state?.current_mask));
      console.log('Mask length:', response.initial_state?.current_mask?.length);
      console.log('First row:', response.initial_state?.current_mask?.[0]);
      
      setGameId(response.game_id);
      setGameState(response.initial_state);
      setImageData(response.image_data);
      setIsPlayerTurn(true);
      setGameConfig(config);
      
      console.log('Game created:', response.game_id);
      
    } catch (err) {
      console.error('Error creating game:', err);
      setError(handleAPIError(err));
    } finally {
      setLoading(false);
    }
  }, [gameConfig]);

  /**
   * Make a move (human player)
   */
  const makeMove = useCallback(async (x, y) => {
    if (!gameId || !gameState) {
      setError('No active game');
      return false;
    }

    if (isGameOver(gameState)) {
      setError('Game is already over');
      return false;
    }

    if (!isPlayerTurn) {
      setError('It\'s not your turn');
      return false;
    }

    // Check if pixel is already revealed
    if (gameState.current_mask && gameState.current_mask[y] && gameState.current_mask[y][x] === 1) {
      setError('This pixel is already revealed');
      return false;
    }

    try {
      setLoading(true);
      setError(null);
      setIsPlayerTurn(false);

      const response = await gameAPI.makeMove(gameId, { x, y });

      if (!response.success) {
        setError(response.error || 'Move failed');
        setIsPlayerTurn(true);
        return false;
      }

      // Update game state
      setGameState(response.game_state);
      
      // If game is over, fetch final result
      if (response.game_over) {
        await fetchGameResult();
        setIsPlayerTurn(false);
      } else {
        // Player can make next move if there's still room and it's their turn
        // In this game, player always goes first, so player turn = even turn numbers
        setIsPlayerTurn(response.game_state.current_turn % 2 === 0 && 
                       response.game_state.current_turn < response.game_state.config.k);
      }

      console.log('Move made:', { x, y }, 'AI move:', response.ai_move);
      return true;

    } catch (err) {
      console.error('Error making move:', err);
      setError(handleAPIError(err));
      setIsPlayerTurn(true);
      return false;
    } finally {
      setLoading(false);
    }
  }, [gameId, gameState, isPlayerTurn, fetchGameResult]);

  /**
   * Fetch current game state
   */
  const fetchGameState = useCallback(async () => {
    if (!gameId) return;

    try {
      const response = await gameAPI.getGameState(gameId);
      setGameState(response.game_state);
      setImageData(response.image_data);
      
      // Update turn state
      if (isGameOver(response.game_state)) {
        setIsPlayerTurn(false);
        await fetchGameResult();
      } else {
        setIsPlayerTurn(response.game_state.current_turn % 2 === 0 && 
                      response.game_state.current_turn < response.game_state.config.k);
      }
      
    } catch (err) {
      console.error('Error fetching game state:', err);
      setError(handleAPIError(err));
    }
  }, [gameId, fetchGameResult]);

  /**
   * Reset game (clean up current game)
   */
  const resetGame = useCallback(() => {
    // Cancel any ongoing requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Clean up old game if exists
    if (gameId) {
      gameAPI.deleteGame(gameId).catch(err => 
        console.warn('Error cleaning up old game:', err)
      );
    }
    
    // Reset all state
    setGameState(null);
    setGameId(null);
    setImageData(null);
    setError(null);
    setGameResult(null);
    setSelectedPixel(null);
    setHoveredPixel(null);
    setIsPlayerTurn(true);
    setLoading(false);
  }, [gameId]);

  /**
   * Update game configuration
   */
  const updateConfig = useCallback((newConfig) => {
    setGameConfig(prev => ({ ...prev, ...newConfig }));
  }, []);

  /**
   * Handle pixel hover
   */
  const handlePixelHover = useCallback((x, y) => {
    if (!gameState || !isPlayerTurn || isGameOver(gameState)) {
      setHoveredPixel(null);
      return;
    }
    
    // Check if current_mask exists and has the right structure
    if (!gameState.current_mask || !Array.isArray(gameState.current_mask) || !gameState.current_mask[y]) {
      console.warn('Invalid mask structure:', gameState.current_mask);
      setHoveredPixel(null);
      return;
    }
    
    // Don't hover over already revealed pixels
    if (gameState.current_mask[y][x] === 1) {
      setHoveredPixel(null);
      return;
    }
    
    setHoveredPixel({ x, y });
  }, [gameState, isPlayerTurn]);

  /**
   * Handle pixel click
   */
  const handlePixelClick = useCallback((x, y) => {
    if (!isPlayerTurn || loading) return;
    
    setSelectedPixel({ x, y });
    makeMove(x, y);
  }, [isPlayerTurn, loading, makeMove]);

  // Cleanup effect
  useEffect(() => {
    const currentAbortController = abortControllerRef.current;
    return () => {
      if (currentAbortController) {
        currentAbortController.abort();
      }
    };
  }, []);

  // Computed values
  const isGameActive = gameState && !isGameOver(gameState);
  const currentTurn = gameState?.current_turn || 0;
  const totalPixels = gameState?.config.k || 0;
  const pixelsRemaining = Math.max(0, totalPixels - currentTurn);

  console.log('useGame hook returning values:', {
    hasGameState: !!gameState,
    loading,
    error,
    gameConfigSet: !!gameConfig
  });

  return {
    // Game state
    gameState,
    gameId,
    imageData,
    gameConfig,
    gameResult,
    
    // UI state
    loading,
    error,
    selectedPixel,
    hoveredPixel,
    isPlayerTurn,
    isGameActive,
    
    // Computed values
    currentTurn,
    totalPixels,
    pixelsRemaining,
    
    // Actions
    createGame,
    makeMove,
    resetGame,
    updateConfig,
    fetchGameState,
    handlePixelHover,
    handlePixelClick,
    
    // Utilities
    clearError: () => setError(null),
  };
};