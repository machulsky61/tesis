import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Zap, Target, Clock } from 'lucide-react';
import { GAME_CONFIG } from '../utils/constants';
import { getPixelColor } from '../utils/helpers';
import { Card, StatsCard } from './ui/Card';

const BoardContainer = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  width: 100%;
  max-width: 800px;
`;

const GameHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  gap: 1rem;
  margin-bottom: 0.5rem;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 1rem;
  }
`;

const TurnIndicatorCard = styled(Card)`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  background: ${props => {
    if (props.loading) return props.theme.background.glass;
    return props.isPlayerTurn 
      ? `linear-gradient(135deg, ${props.theme.semantic.success}20, ${props.theme.semantic.success}10)`
      : `linear-gradient(135deg, ${props.theme.semantic.error}20, ${props.theme.semantic.error}10)`;
  }};
  border: 2px solid ${props => {
    if (props.loading) return props.theme.brand.primary + '40';
    return props.isPlayerTurn 
      ? props.theme.semantic.success + '60'
      : props.theme.semantic.error + '60';
  }};
  color: ${props => {
    if (props.loading) return props.theme.text.primary;
    return props.isPlayerTurn 
      ? props.theme.semantic.success
      : props.theme.semantic.error;
  }};
  font-weight: 600;
  font-size: 1rem;
  text-align: center;
  min-width: 280px;
  
  .turn-icon {
    width: 1.25rem;
    height: 1.25rem;
    ${props => props.loading && 'animation: spin 1s linear infinite;'}
  }

  .turn-text {
    white-space: nowrap;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  @media (max-width: 640px) {
    min-width: auto;
    flex: 1;
  }
`;

const PreviewSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  
  .preview-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: ${props => props.theme.text.secondary};
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
`;

const GridContainer = styled(motion.div)`
  position: relative;
  width: 100%;
  max-width: 640px;
  aspect-ratio: 1;
  border-radius: ${props => props.theme.radius.xl};
  overflow: hidden;
  background: ${props => props.theme.background.surface};
  box-shadow: ${props => props.theme.shadows.xl};
  border: 1px solid ${props => props.theme.background.glass};

  /* Disabled state when loading */
  ${props => props.disabled && `
    pointer-events: none;
    opacity: 0.8;
  `}

  @media (max-width: 768px) {
    max-width: 100%;
  }
`;

const BackgroundImage = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: repeat(${GAME_CONFIG.GRID_SIZE}, 1fr);
  grid-template-rows: repeat(${GAME_CONFIG.GRID_SIZE}, 1fr);
  background: ${props => props.theme.game.grid};
`;

const BackgroundPixel = styled.div`
  width: 100%;
  height: 100%;
  border: 1px solid ${props => props.theme.game.grid};
  box-sizing: border-box;
`;

const GridOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: repeat(${GAME_CONFIG.GRID_SIZE}, 1fr);
  grid-template-rows: repeat(${GAME_CONFIG.GRID_SIZE}, 1fr);
  z-index: 10;
`;

const Pixel = styled(motion.div)`
  width: 100%;
  height: 100%;
  cursor: ${props => props.canClick ? 'pointer' : props.isRevealed ? 'default' : 'not-allowed'};
  position: relative;
  border: 1px solid ${props => props.theme.game.grid};
  box-sizing: border-box;
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);
  
  /* Base states - use semi-transparent overlays instead of solid backgrounds */
  ${props => !props.isRevealed && !props.canClick && `
    background-color: ${props.theme.game.unrevealed};
    opacity: 0.4; /* Slightly more visible */
    
    /* Overlay to dim unavailable pixels */
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: ${props.theme.background.surface};
      opacity: 0.7; /* Dim but still show underlying image */
      pointer-events: none;
      z-index: 1;
    }
  `}
  
  ${props => !props.isRevealed && props.canClick && `
    background-color: transparent; /* Show underlying image */
    opacity: 0.8; /* More visible for clickable pixels */
    
    /* Light overlay to indicate it's selectable */
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: ${props.theme.game.unrevealed};
      opacity: 0.3; /* Light overlay */
      pointer-events: none;
      z-index: 1;
      transition: opacity 200ms ease;
    }
    
    &:hover {
      opacity: 1;
      transform: scale(1.02);
      box-shadow: ${props.theme.shadows.md};
      z-index: 15;
      border-color: ${props.theme.game.hover};
      
      &::before {
        background: ${props.theme.game.hover};
        opacity: 0.4;
      }
    }
  `}
  
  ${props => props.isRevealed && `
    background-color: transparent; /* Keep background transparent to show underlying image */
    opacity: 1;
    border: 3px solid ${props.revealedBy === 'human' 
      ? props.theme.game.humanMove 
      : props.theme.game.aiMove
    };
    box-shadow: 
      0 0 0 1px ${props.revealedBy === 'human' 
        ? props.theme.game.humanMove + '60'
        : props.theme.game.aiMove + '60'
      },
      ${props.theme.shadows.md};
    
    /* Subtle overlay to indicate reveal status without hiding image */
    &::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: ${props.revealedBy === 'human' 
        ? props.theme.game.humanMove + '08'  /* Very subtle overlay */
        : props.theme.game.aiMove + '08'
      };
      pointer-events: none;
      z-index: 1;
      border-radius: 1px;
    }
  `}
  
  ${props => props.isHovered && props.canClick && `
    background-color: ${props.theme.game.hover} !important;
    opacity: 0.9 !important;
    transform: scale(1.05) !important;
    z-index: 20 !important;
    border-color: ${props.theme.brand.primary} !important;
    box-shadow: 
      0 0 20px ${props.theme.brand.primary}40,
      ${props.theme.shadows.lg} !important;
  `}
`;

const GameStats = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
  width: 100%;
  margin-top: 1rem;

  @media (max-width: 640px) {
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
  }
`;

const PreviewImage = styled(Card)`
  width: 120px;
  height: 120px;
  display: grid;
  grid-template-columns: repeat(28, 1fr);
  grid-template-rows: repeat(28, 1fr);
  border: 1px solid ${props => props.theme.background.glass};
  padding: 0.5rem;
  opacity: 0.8;
  transition: all 300ms ease;
  
  &:hover {
    opacity: 1;
    transform: scale(1.02);
  }
  
  .preview-pixel {
    width: 100%;
    height: 100%;
    border-radius: 1px;
  }

  @media (max-width: 640px) {
    width: 100px;
    height: 100px;
  }
`;

const MoveLegend = styled.div`
  display: flex;
  justify-content: center;
  gap: 1.5rem;
  flex-wrap: wrap;
  
  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: ${props => props.theme.text.secondary};
    
    .legend-color {
      width: 1.25rem;
      height: 1.25rem;
      border-radius: ${props => props.theme.radius.sm};
      border: 2px solid;
    }
    
    &.unrevealed .legend-color {
      background: ${props => props.theme.game.unrevealed};
      border-color: ${props => props.theme.text.muted};
    }
    
    &.human .legend-color {
      background: ${props => props.theme.game.humanMove}20;
      border-color: ${props => props.theme.game.humanMove};
    }
    
    &.ai .legend-color {
      background: ${props => props.theme.game.aiMove}20;
      border-color: ${props => props.theme.game.aiMove};
    }
  }

  @media (max-width: 640px) {
    gap: 1rem;
    
    .legend-item {
      font-size: 0.75rem;
      
      .legend-color {
        width: 1rem;
        height: 1rem;
      }
    }
  }
`;

const LoadingOverlay = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: ${props => props.theme.background.overlay};
  backdrop-filter: ${props => props.theme.blur.md};
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  border-radius: ${props => props.theme.radius.xl};

  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    background: ${props => props.theme.background.surface};
    padding: 2rem;
    border-radius: ${props => props.theme.radius.lg};
    box-shadow: ${props => props.theme.shadows.xl};
    border: 1px solid ${props => props.theme.background.glass};

    .loading-spinner {
      width: 3rem;
      height: 3rem;
      border: 3px solid ${props => props.theme.interactive.disabled};
      border-radius: 50%;
      border-top-color: ${props => props.theme.brand.primary};
      animation: spin 1s linear infinite;
    }

    .loading-text {
      color: ${props => props.theme.text.primary};
      font-size: 1rem;
      font-weight: 600;
      text-align: center;
    }

    .loading-subtext {
      color: ${props => props.theme.text.secondary};
      font-size: 0.875rem;
      text-align: center;
    }

    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  }
`;

const GameBoard = ({
  gameState,
  imageData,
  hoveredPixel,
  selectedPixel,
  isPlayerTurn,
  onPixelClick,
  onPixelHover,
  loading
}) => {
  // Debug and validation
  console.log('GameBoard render:', { 
    hasGameState: !!gameState, 
    hasImageData: !!imageData,
    isPlayerTurn,
    loading
  });

  // Calculate image statistics for contrast stretching
  let imageStats = null;
  if (imageData && Array.isArray(imageData)) {
    const allValues = [];
    imageData.forEach(row => {
      if (Array.isArray(row)) {
        row.forEach(val => allValues.push(val));
      }
    });
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const avgVal = allValues.reduce((a, b) => a + b, 0) / allValues.length;
    imageStats = { minVal, maxVal, avgVal };
  }

  // Validation checks
  if (!gameState) {
    return (
      <BoardContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Card padding="2rem">
          <h3>No game state available</h3>
          <p>Please start a new game to begin.</p>
        </Card>
      </BoardContainer>
    );
  }

  if (!imageData || !Array.isArray(imageData) || imageData.length !== 28) {
    return (
      <BoardContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Card padding="2rem">
          <h3>Invalid image data</h3>
          <p>Backend connection issue or invalid image format.</p>
        </Card>
      </BoardContainer>
    );
  }

  const { current_mask, moves, current_turn, config, judge_prediction } = gameState;

  // Validate current_mask structure
  if (!current_mask || !Array.isArray(current_mask) || current_mask.length !== 28) {
    return (
      <BoardContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Card padding="2rem">
          <h3>Invalid mask data</h3>
          <p>Game state is corrupted. Please restart.</p>
        </Card>
      </BoardContainer>
    );
  }

  // Create pixel revelation map
  const pixelRevealedBy = {};
  moves?.forEach(move => {
    const key = `${move.pixel.x},${move.pixel.y}`;
    pixelRevealedBy[key] = move.player_type;
  });

  // Check if pixel is available (above threshold)
  const isPixelAvailable = (pixelValue, allowZero = config?.allow_zero_pixels) => {
    const threshold = allowZero ? 0.0 : 0.001;
    return typeof pixelValue === 'number' && pixelValue >= threshold;
  };

  // Enhanced pixel interaction handlers
  const handlePixelClick = (x, y) => {
    if (!isPlayerTurn || loading || current_mask[y][x] === 1) return;
    
    const pixelValue = imageData[y][x];
    if (!isPixelAvailable(pixelValue)) {
      console.warn(`Pixel (${x}, ${y}) below threshold: ${pixelValue}`);
      return;
    }
    
    onPixelClick(x, y);
  };

  const handlePixelHover = (x, y, isEntering) => {
    if (x < 0 || x >= 28 || y < 0 || y >= 28) return;
    if (!current_mask?.[y] || typeof current_mask[y][x] === 'undefined') return;
    if (!isPlayerTurn || loading || current_mask[y][x] === 1) {
      onPixelHover(null);
      return;
    }
    
    const pixelValue = imageData[y][x];
    if (!isPixelAvailable(pixelValue)) {
      onPixelHover(null);
      return;
    }
    
    onPixelHover(isEntering ? { x, y } : null);
  };

  // Render background pixel with enhanced contrast
  const renderBackgroundPixel = (x, y) => {
    const pixelValue = imageData[y][x];
    const normalizedValue = typeof pixelValue === 'number' ? Math.max(0, Math.min(1, pixelValue)) : 0;
    
    // Apply contrast stretching for better visibility
    let intensity;
    if (imageStats && imageStats.maxVal > imageStats.minVal) {
      const stretched = (normalizedValue - imageStats.minVal) / (imageStats.maxVal - imageStats.minVal);
      intensity = Math.floor(stretched * 255);
    } else {
      intensity = Math.floor(normalizedValue * 255);
    }
    
    return (
      <BackgroundPixel
        key={`bg-${x}-${y}`}
        style={{
          backgroundColor: `rgb(${intensity}, ${intensity}, ${intensity})`,
        }}
      />
    );
  };

  // Render interactive overlay pixel with animations
  const renderOverlayPixel = (x, y) => {
    const isRevealed = current_mask[y][x] === 1;
    const pixelValue = imageData[y][x];
    const revealedBy = pixelRevealedBy[`${x},${y}`];
    const isHovered = hoveredPixel && hoveredPixel.x === x && hoveredPixel.y === y;
    const isSelected = selectedPixel && selectedPixel.x === x && selectedPixel.y === y;
    const isAvailable = isPixelAvailable(pixelValue);
    const canClick = isPlayerTurn && !loading && !isRevealed && isAvailable;

    return (
      <Pixel
        key={`overlay-${x}-${y}`}
        canClick={canClick}
        isRevealed={isRevealed}
        isHovered={isHovered}
        isSelected={isSelected}
        revealedBy={revealedBy}
        onClick={() => handlePixelClick(x, y)}
        onMouseEnter={() => handlePixelHover(x, y, true)}
        onMouseLeave={() => handlePixelHover(x, y, false)}
        title={isRevealed 
          ? `Revealed by ${revealedBy} • Value: ${pixelValue?.toFixed(3) || 'N/A'}`
          : canClick
            ? `Click to reveal • Value: ${pixelValue?.toFixed(3) || 'N/A'}`
            : `Below threshold • Value: ${pixelValue?.toFixed(3) || 'N/A'}`
        }
        // Framer Motion animations
        initial={isRevealed ? { scale: 0.8, opacity: 0 } : false}
        animate={isRevealed ? { 
          scale: 1, 
          opacity: 1,
          transition: { 
            type: "spring",
            stiffness: 500,
            damping: 30,
            delay: 0.1
          }
        } : {}}
        whileHover={canClick ? {
          scale: 1.02,
          transition: { duration: 0.1 }
        } : {}}
        whileTap={canClick ? {
          scale: 0.98,
          transition: { duration: 0.1 }
        } : {}}
      />
    );
  };

  // Generate pixel arrays
  const backgroundPixels = [];
  const overlayPixels = [];
  
  for (let y = 0; y < GAME_CONFIG.GRID_SIZE; y++) {
    for (let x = 0; x < GAME_CONFIG.GRID_SIZE; x++) {
      backgroundPixels.push(renderBackgroundPixel(x, y));
      overlayPixels.push(renderOverlayPixel(x, y));
    }
  }

  // Generate preview image pixels
  const renderPreviewPixel = (x, y) => {
    const pixelValue = imageData[y][x];
    const normalizedValue = typeof pixelValue === 'number' ? Math.max(0, Math.min(1, pixelValue)) : 0;
    
    let intensity;
    if (imageStats && imageStats.maxVal > imageStats.minVal) {
      const stretched = (normalizedValue - imageStats.minVal) / (imageStats.maxVal - imageStats.minVal);
      intensity = Math.floor(stretched * 255);
    } else {
      intensity = Math.floor(normalizedValue * 255);
    }
    
    return (
      <div
        key={`preview-${x}-${y}`}
        className="preview-pixel"
        style={{ backgroundColor: `rgb(${intensity}, ${intensity}, ${intensity})` }}
      />
    );
  };

  const previewPixels = [];
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      previewPixels.push(renderPreviewPixel(x, y));
    }
  }

  // Turn indicator content
  const getTurnIcon = () => {
    if (loading) return <Zap className="turn-icon" />;
    return isPlayerTurn ? <Target className="turn-icon" /> : <Eye className="turn-icon" />;
  };

  const getTurnText = () => {
    if (loading) return gameState ? 'Processing move...' : 'Creating game...';
    return isPlayerTurn ? 'Your Turn - Click a pixel' : 'AI is thinking...';
  };

  return (
    <BoardContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      {/* Game Header */}
      <GameHeader>
        <TurnIndicatorCard 
          isPlayerTurn={isPlayerTurn} 
          loading={loading}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={loading ? 'loading' : isPlayerTurn ? 'player' : 'ai'}
              className="status-content"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
            >
              {getTurnIcon()}
              <span className="turn-text">{getTurnText()}</span>
            </motion.div>
          </AnimatePresence>
        </TurnIndicatorCard>

        <PreviewSection>
          <div className="preview-label">Original Image</div>
          <PreviewImage
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            {previewPixels}
          </PreviewImage>
        </PreviewSection>
      </GameHeader>

      {/* Main Game Grid */}
      <GridContainer
        disabled={loading}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <BackgroundImage>{backgroundPixels}</BackgroundImage>
        <GridOverlay>{overlayPixels}</GridOverlay>
        
        {/* Loading Overlay - only during move processing */}
        <AnimatePresence>
          {loading && gameState && (
            <LoadingOverlay
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="loading-content">
                <div className="loading-spinner" />
                <div className="loading-text">
                  {isPlayerTurn ? 'Processing your move...' : 'AI is thinking...'}
                </div>
                <div className="loading-subtext">
                  Please wait while the move is being processed
                </div>
              </div>
            </LoadingOverlay>
          )}
        </AnimatePresence>
      </GridContainer>

      {/* Game Statistics */}
      <GameStats>
        <StatsCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
        >
          <div className="label">Progress</div>
          <div className="value">{current_turn} / {config?.k || 0}</div>
        </StatsCard>

        <StatsCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.5 }}
        >
          <div className="label">Remaining</div>
          <div className="value">{(config?.k || 0) - current_turn}</div>
        </StatsCard>

        {judge_prediction && (
          <>
            <StatsCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.6 }}
            >
              <div className="label">Judge Thinks</div>
              <div className="value">{judge_prediction.predicted_class}</div>
            </StatsCard>

            <StatsCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.7 }}
            >
              <div className="label">Confidence</div>
              <div className="value">{(judge_prediction.confidence * 100).toFixed(1)}%</div>
            </StatsCard>
          </>
        )}
      </GameStats>

      {/* Move Legend */}
      <MoveLegend>
        <div className="legend-item unrevealed">
          <div className="legend-color" />
          <span>Unrevealed</span>
        </div>
        <div className="legend-item human">
          <div className="legend-color" />
          <span>Your moves</span>
        </div>
        <div className="legend-item ai">
          <div className="legend-color" />
          <span>AI moves</span>
        </div>
      </MoveLegend>
    </BoardContainer>
  );
};

export default GameBoard;