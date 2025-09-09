// Add error checking for imports
console.log('Importing constants...');

let VALIDATION, ERROR_MESSAGES, DEFAULT_GAME_CONFIG;

try {
  const constants = require('./constants');
  VALIDATION = constants.VALIDATION;
  ERROR_MESSAGES = constants.ERROR_MESSAGES;
  DEFAULT_GAME_CONFIG = constants.DEFAULT_GAME_CONFIG;
  console.log('Constants imported successfully:', { DEFAULT_GAME_CONFIG });
} catch (error) {
  console.error('Error importing constants:', error);
  // Fallback values
  DEFAULT_GAME_CONFIG = {
    k: 6,
    player_role: 'honest',
    agent_type: 'greedy',
    precommit: false,
    judge_name: '28',
    seed: null,
  };
  VALIDATION = {
    pixelCoordinate: { min: 0, max: 27 },
    gameConfig: { k: { min: 2, max: 12 } }
  };
  ERROR_MESSAGES = {
    API_ERROR: 'Something went wrong',
    NETWORK_ERROR: 'Network error'
  };
}

// Re-export for useGame hook
export { DEFAULT_GAME_CONFIG };

/**
 * Validate pixel coordinates
 */
export const validatePixelCoordinate = (x, y) => {
  const { min, max } = VALIDATION.pixelCoordinate;
  return x >= min && x <= max && y >= min && y <= max;
};

/**
 * Validate game configuration
 */
export const validateGameConfig = (config) => {
  const errors = [];

  if (!config) {
    errors.push('Configuration is required');
    return errors;
  }

  // Validate k (number of pixels)
  if (config.k < VALIDATION.gameConfig.k.min || config.k > VALIDATION.gameConfig.k.max) {
    errors.push(`Number of pixels must be between ${VALIDATION.gameConfig.k.min} and ${VALIDATION.gameConfig.k.max}`);
  }

  // Validate player_role
  if (!['honest', 'liar'].includes(config.player_role)) {
    errors.push('Player role must be either "honest" or "liar"');
  }

  // Validate agent_type
  if (!['greedy', 'mcts'].includes(config.agent_type)) {
    errors.push('Agent type must be either "greedy" or "mcts"');
  }

  // Validate judge_name
  if (!config.judge_name || typeof config.judge_name !== 'string') {
    errors.push('Judge name is required');
  }

  return errors;
};

/**
 * Format time duration in human readable format
 */
export const formatDuration = (seconds) => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};

/**
 * Format percentage
 */
export const formatPercentage = (value, decimals = 1) => {
  return `${(value * 100).toFixed(decimals)}%`;
};

/**
 * Get pixel color based on state and who revealed it
 */
export const getPixelColor = (isRevealed, revealedBy, isHovered = false, pixelValue = 0) => {
  if (isHovered && !isRevealed) {
    return '#ffeb3b'; // Yellow for hover
  }
  
  if (!isRevealed) {
    return '#e0e0e0'; // Gray for unrevealed
  }

  // Debug logging for first few pixels
  if (Math.random() < 0.01) { // Log only 1% to avoid spam
    console.log('Pixel render:', { isRevealed, revealedBy, pixelValue, pixelValueType: typeof pixelValue });
  }

  // Ensure pixelValue is valid and in correct range
  let normalizedValue = pixelValue;
  
  // Handle different possible data formats
  if (typeof pixelValue !== 'number' || isNaN(pixelValue)) {
    normalizedValue = 0;
  } else if (pixelValue > 1) {
    // If pixel value is in 0-255 range, normalize to 0-1
    normalizedValue = pixelValue / 255;
  }
  
  // Ensure value is between 0 and 1
  normalizedValue = Math.max(0, Math.min(1, normalizedValue));
  
  // Convert to 0-255 intensity (invert for MNIST: 0=black, 1=white)
  const intensity = Math.floor((1 - normalizedValue) * 255); // Inverted for MNIST
  const grayColor = `rgb(${intensity}, ${intensity}, ${intensity})`;
  
  const borderColor = revealedBy === 'human' ? '#4caf50' : '#f44336';
  
  return {
    backgroundColor: grayColor,
    border: `2px solid ${borderColor}`,
    boxSizing: 'border-box'
  };
};

/**
 * Create a 2D array initialized with a value
 */
export const create2DArray = (rows, cols, initialValue = 0) => {
  return Array(rows).fill().map(() => Array(cols).fill(initialValue));
};

/**
 * Deep clone an object
 */
export const deepClone = (obj) => {
  return JSON.parse(JSON.stringify(obj));
};

/**
 * Calculate game statistics from moves
 */
export const calculateGameStats = (moves) => {
  const humanMoves = moves.filter(move => move.player_type === 'human');
  const aiMoves = moves.filter(move => move.player_type === 'ai');

  // Calculate average positions
  const avgHumanX = humanMoves.length > 0 
    ? humanMoves.reduce((sum, move) => sum + move.pixel.x, 0) / humanMoves.length 
    : 0;
  const avgHumanY = humanMoves.length > 0 
    ? humanMoves.reduce((sum, move) => sum + move.pixel.y, 0) / humanMoves.length 
    : 0;

  const avgAIX = aiMoves.length > 0 
    ? aiMoves.reduce((sum, move) => sum + move.pixel.x, 0) / aiMoves.length 
    : 0;
  const avgAIY = aiMoves.length > 0 
    ? aiMoves.reduce((sum, move) => sum + move.pixel.y, 0) / aiMoves.length 
    : 0;

  // Calculate move timing (if timestamps available)
  const moveTimes = moves.map((move, index) => {
    if (index === 0) return 0;
    return move.timestamp - moves[index - 1].timestamp;
  }).filter(time => time > 0);

  const avgMoveTime = moveTimes.length > 0 
    ? moveTimes.reduce((sum, time) => sum + time, 0) / moveTimes.length 
    : 0;

  return {
    humanMoves: humanMoves.length,
    aiMoves: aiMoves.length,
    avgHumanPosition: { x: avgHumanX, y: avgHumanY },
    avgAIPosition: { x: avgAIX, y: avgAIY },
    avgMoveTime,
    totalMoves: moves.length,
  };
};

/**
 * Get strategy description based on average positions
 */
export const getStrategyDescription = (avgPosition, gridSize = 28) => {
  const centerX = gridSize / 2;
  const centerY = gridSize / 2;
  const { x, y } = avgPosition;

  if (Math.abs(x - centerX) < 3 && Math.abs(y - centerY) < 3) {
    return 'Center-focused';
  } else if (x < gridSize * 0.3 || x > gridSize * 0.7) {
    return 'Edge-focused';
  } else if (y < gridSize * 0.3 || y > gridSize * 0.7) {
    return 'Top/Bottom-focused';
  } else {
    return 'Distributed';
  }
};

/**
 * Handle API errors consistently
 */
export const handleAPIError = (error) => {
  console.error('API Error:', error);

  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    
    switch (status) {
      case 404:
        return ERROR_MESSAGES.GAME_NOT_FOUND;
      case 400:
        return (data && data.detail) || ERROR_MESSAGES.INVALID_MOVE;
      case 500:
        return ERROR_MESSAGES.API_ERROR;
      default:
        return (data && data.detail) || ERROR_MESSAGES.API_ERROR;
    }
  } else if (error.request) {
    // Network error - backend is not running
    return 'Cannot connect to backend server. Make sure it\'s running on http://localhost:8000';
  } else {
    // Something else
    return error.message || ERROR_MESSAGES.API_ERROR;
  }
};

/**
 * Generate a random seed for game reproducibility
 */
export const generateSeed = () => {
  return Math.floor(Math.random() * 1000000);
};

/**
 * Local storage helpers
 */
export const storage = {
  get: (key, defaultValue = null) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return defaultValue;
    }
  },

  set: (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error writing to localStorage:', error);
    }
  },

  remove: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from localStorage:', error);
    }
  },
};

/**
 * Debounce function for performance optimization
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Throttle function for performance optimization  
 */
export const throttle = (func, limit) => {
  let inThrottle;
  return function executedFunction(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

/**
 * Check if the game is over
 */
export const isGameOver = (gameState) => {
  return gameState.status === 'completed' || 
         gameState.status === 'abandoned' ||
         gameState.current_turn >= gameState.config.k;
};

/**
 * Get human-readable role description
 */
export const getRoleDescription = (role) => {
  return role === 'honest' 
    ? 'Try to help the judge classify correctly'
    : 'Try to mislead the judge into wrong classification';
};

/**
 * Get agent type description
 */
export const getAgentDescription = (agentType) => {
  return agentType === 'greedy'
    ? 'Fast agent that picks pixels greedily'
    : 'Strategic agent that plans ahead (slower but smarter)';
};