// Game constants
export const GAME_CONFIG = {
  GRID_SIZE: 28,
  MIN_PIXELS: 2,
  MAX_PIXELS: 12,
  DEFAULT_PIXELS: 6,
};

export const PLAYER_ROLES = {
  HONEST: 'honest',
  LIAR: 'liar',
};

export const AGENT_TYPES = {
  GREEDY: 'greedy',
  MCTS: 'mcts',
};

export const GAME_STATUS = {
  WAITING: 'waiting',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  ABANDONED: 'abandoned',
};

export const PLAYER_TYPES = {
  HUMAN: 'human',
  AI: 'ai',
};

// UI constants
export const COLORS = {
  primary: '#667eea',
  secondary: '#764ba2',
  success: '#4caf50',
  error: '#f44336',
  warning: '#ff9800',
  info: '#2196f3',
  
  // Pixel states
  unrevealed: '#e0e0e0',
  revealed: '#ffffff',
  humanMove: '#4caf50',
  aiMove: '#f44336',
  hover: '#ffeb3b',
  
  // Background
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  cardBackground: 'rgba(255, 255, 255, 0.95)',
  overlayBackground: 'rgba(0, 0, 0, 0.5)',
};

export const ANIMATIONS = {
  pixelReveal: 'pixelReveal 0.3s ease-in-out',
  slideIn: 'slideIn 0.5s ease-out',
  fadeIn: 'fadeIn 0.3s ease-in',
  bounce: 'bounce 0.6s ease-in-out',
};

// Default game configuration
export const DEFAULT_GAME_CONFIG = {
  k: 6,
  player_role: PLAYER_ROLES.HONEST,
  agent_type: AGENT_TYPES.GREEDY,
  precommit: false,
  judge_name: '28',
  seed: null,
  allow_zero_pixels: false, // true = OOD mode (allow black pixels), false = standard (only bright pixels)
};

// Validation rules
export const VALIDATION = {
  pixelCoordinate: {
    min: 0,
    max: 27,
  },
  gameConfig: {
    k: {
      min: GAME_CONFIG.MIN_PIXELS,
      max: GAME_CONFIG.MAX_PIXELS,
    },
  },
};

// Error messages
export const ERROR_MESSAGES = {
  GAME_NOT_FOUND: 'Game not found',
  INVALID_MOVE: 'Invalid move - pixel already revealed',
  GAME_NOT_IN_PROGRESS: 'Game is not in progress',
  API_ERROR: 'Something went wrong. Please try again.',
  NETWORK_ERROR: 'Network error. Please check your connection.',
  PIXEL_ALREADY_REVEALED: 'This pixel is already revealed',
  GAME_OVER: 'Game is already over',
};

// Success messages
export const SUCCESS_MESSAGES = {
  GAME_CREATED: 'Game created successfully!',
  MOVE_MADE: 'Move made successfully!',
  GAME_COMPLETED: 'Game completed!',
};

// Tutorial steps (for future implementation)
export const TUTORIAL_STEPS = {
  WELCOME: 'welcome',
  CONFIG: 'config',
  GRID: 'grid',
  FIRST_MOVE: 'first_move',
  AI_TURN: 'ai_turn',
  JUDGE_PREDICTION: 'judge_prediction',
  GAME_END: 'game_end',
};

// Local storage keys
export const STORAGE_KEYS = {
  GAME_CONFIG: 'debate_game_config',
  TUTORIAL_COMPLETED: 'debate_game_tutorial_completed',
  STATS: 'debate_game_stats',
  PREFERENCES: 'debate_game_preferences',
};

// CSS class names
export const CSS_CLASSES = {
  pixel: 'pixel',
  pixelRevealed: 'pixel-revealed',
  pixelUnrevealed: 'pixel-unrevealed',
  pixelHuman: 'pixel-human',
  pixelAI: 'pixel-ai',
  pixelHover: 'pixel-hover',
  gameGrid: 'game-grid',
  gameCard: 'game-card',
  configPanel: 'config-panel',
  statsPanel: 'stats-panel',
};