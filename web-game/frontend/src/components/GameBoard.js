import React from 'react';
import styled from 'styled-components';
import { COLORS, GAME_CONFIG } from '../utils/constants';
import { getPixelColor } from '../utils/helpers';

const BoardContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 20px;
`;

const GridContainer = styled.div`
  position: relative;
  max-width: 560px;
  max-height: 560px;
  width: 100%;
  aspect-ratio: 1;
  border: 3px solid #333;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
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
  background: black; /* MNIST background */
`;

const BackgroundPixel = styled.div`
  width: 100%;
  height: 100%;
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

const Pixel = styled.div`
  width: 100%;
  height: 100%;
  cursor: ${props => props.canClick ? 'pointer' : props.isRevealed ? 'default' : 'not-allowed'};
  transition: all 0.2s ease;
  position: relative;
  border: 1px solid transparent; /* Back to 1px as requested */
  box-sizing: border-box; /* Include border in size calculations */
  
  /* Overlay states - TRANSPARENT backgrounds to show underlying image */
  ${props => !props.isRevealed && !props.canClick && `
    background-color: transparent;
    border: 1px solid rgba(100, 100, 100, 0.8); /* 1px border for unavailable pixels */
  `}
  
  ${props => !props.isRevealed && props.canClick && `
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 0.3); /* 1px border for available pixels */
  `}
  
  ${props => props.isRevealed && `
    background-color: transparent; /* Show underlying image */
    border: 1px solid ${props.revealedBy === 'human' ? '#4CAF50' : '#F44336'};
    box-shadow: inset 0 0 0 1px ${props.revealedBy === 'human' ? '#4CAF50' : '#F44336'};
    outline: 1px solid ${props.revealedBy === 'human' ? '#4CAF50' : '#F44336'};
    outline-offset: -1px; /* Outline inside the element */
  `}
  
  ${props => props.isHovered && props.canClick && `
    background-color: rgba(255, 235, 59, 0.3) !important; /* Reduced opacity */
    border: 1px solid #FFC107 !important; /* Keep 1px consistency */
    outline: 1px solid #FFC107 !important; /* Extra emphasis with outline */
    outline-offset: -1px;
    transform: scale(1.05);
    z-index: 15;
    box-shadow: 0 0 8px rgba(255, 235, 59, 0.6); /* Reduced shadow */
  `}
  
  &:hover {
    ${props => props.canClick && `
      transform: scale(1.02);
      z-index: 12;
    `}
  }
`;

const TurnIndicator = styled.div`
  background: ${props => props.isPlayerTurn ? COLORS.success : COLORS.error};
  color: white;
  padding: 12px 24px;
  border-radius: 25px;
  font-weight: bold;
  font-size: 16px;
  text-align: center;
  min-width: 200px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  animation: ${props => props.isPlayerTurn ? 'pulse 1.5s ease-in-out infinite alternate' : 'none'};

  @keyframes pulse {
    from { transform: scale(1); }
    to { transform: scale(1.05); }
  }
`;

const GameStats = styled.div`
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 10px;
`;

const StatItem = styled.div`
  background: ${COLORS.cardBackground};
  padding: 12px 20px;
  border-radius: 8px;
  text-align: center;
  min-width: 120px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  
  .label {
    font-size: 12px;
    color: #666;
    margin-bottom: 4px;
    text-transform: uppercase;
    font-weight: 600;
  }
  
  .value {
    font-size: 18px;
    font-weight: bold;
    color: #333;
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
  console.log('GameBoard props:', { 
    hasGameState: !!gameState, 
    hasImageData: !!imageData,
    imageDataType: typeof imageData,
    imageDataLength: Array.isArray(imageData) ? imageData.length : 'not array'
  });

  // Debug image data values to understand the range
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
    console.log('Image data analysis:', { minVal, maxVal, avgVal, sampleValues: allValues.slice(0, 10) });
  }

  if (!gameState) {
    return (
      <BoardContainer>
        <div>No game state available.</div>
      </BoardContainer>
    );
  }

  if (!imageData) {
    return (
      <BoardContainer>
        <div>No image data available. Backend may not be responding correctly.</div>
        <div style={{fontSize: '12px', opacity: 0.7, marginTop: '10px'}}>
          Check Debug panel for backend connection status.
        </div>
      </BoardContainer>
    );
  }

  // Validate image data structure
  if (!Array.isArray(imageData) || imageData.length !== 28 || !Array.isArray(imageData[0]) || imageData[0].length !== 28) {
    return (
      <BoardContainer>
        <div>Invalid image data format received from backend.</div>
        <div style={{fontSize: '12px', marginTop: '10px'}}>
          Expected 28x28 array, got: {typeof imageData} {Array.isArray(imageData) ? `[${imageData.length}]` : ''}
        </div>
      </BoardContainer>
    );
  }

  const { current_mask, moves, current_turn, config, judge_prediction } = gameState;

  // Validate current_mask structure
  if (!current_mask || !Array.isArray(current_mask) || current_mask.length !== 28) {
    return (
      <BoardContainer>
        <div>Invalid game mask data.</div>
        <div style={{fontSize: '12px', marginTop: '10px'}}>
          Mask type: {typeof current_mask}, length: {Array.isArray(current_mask) ? current_mask.length : 'N/A'}
        </div>
      </BoardContainer>
    );
  }

  // Create a map of who revealed each pixel
  const pixelRevealedBy = {};
  moves.forEach(move => {
    const key = `${move.pixel.x},${move.pixel.y}`;
    pixelRevealedBy[key] = move.player_type;
  });

  // Handle pixel interactions
  const handlePixelClick = (x, y) => {
    if (!isPlayerTurn || loading || current_mask[y][x] === 1) return;
    
    // Check if pixel is available (respects threshold)
    const pixelValue = imageData[y][x];
    if (!isPixelAvailable(pixelValue)) {
      console.warn(`Pixel (${x}, ${y}) with value ${pixelValue} is below threshold`);
      return;
    }
    
    onPixelClick(x, y);
  };

  const handlePixelHover = (x, y, isEntering) => {
    // Validate coordinates
    if (x < 0 || x >= 28 || y < 0 || y >= 28) {
      console.warn('Invalid pixel coordinates:', { x, y });
      return;
    }
    
    // Validate mask structure
    if (!current_mask || !current_mask[y] || typeof current_mask[y][x] === 'undefined') {
      console.warn('Invalid mask at coordinates:', { x, y, mask_row: current_mask?.[y] });
      return;
    }
    
    if (!isPlayerTurn || loading || current_mask[y][x] === 1) {
      onPixelHover(null);
      return;
    }
    
    // Check if pixel is available (respects threshold)
    const pixelValue = imageData[y][x];
    if (!isPixelAvailable(pixelValue)) {
      onPixelHover(null);
      return;
    }
    
    onPixelHover(isEntering ? { x, y } : null);
  };

  // Check if pixel is available (above or equal to threshold)
  const isPixelAvailable = (pixelValue, allowZero = config.allow_zero_pixels) => {
    const threshold = allowZero ? 0.0 : 0.001;
    return typeof pixelValue === 'number' && pixelValue >= threshold;
  };

  // Render background pixel (always visible) - original MNIST image
  const renderBackgroundPixel = (x, y) => {
    const pixelValue = imageData[y][x];
    
    // Debug logging for a few pixels to understand the values
    if ((x === 0 && y === 0) || (x === 14 && y === 14) || (x === 10 && y === 10)) {
      console.log(`Pixel (${x},${y}): raw=${pixelValue}, type=${typeof pixelValue}`);
    }
    
    const normalizedValue = typeof pixelValue === 'number' ? Math.max(0, Math.min(1, pixelValue)) : 0;
    
    // Debug the normalized value too
    if ((x === 0 && y === 0) || (x === 14 && y === 14) || (x === 10 && y === 10)) {
      console.log(`Pixel (${x},${y}): normalized=${normalizedValue}, intensity will be=${Math.floor(normalizedValue * 255)}`);
    }
    
    // Apply contrast stretching to use full black-white range
    let intensity;
    if (imageStats && imageStats.maxVal > imageStats.minVal) {
      // Stretch the values to use full 0-255 range
      const stretched = (normalizedValue - imageStats.minVal) / (imageStats.maxVal - imageStats.minVal);
      intensity = Math.floor(stretched * 255);
    } else {
      // Fallback to normal mapping
      intensity = Math.floor(normalizedValue * 255);
    }
    
    // Debug for specific pixels
    if ((x === 0 && y === 0) || (x === 14 && y === 14) || (x === 10 && y === 10)) {
      console.log(`Pixel (${x},${y}): original=${normalizedValue}, stretched_intensity=${intensity}`);
    }
    
    return (
      <BackgroundPixel
        key={`bg-${x}-${y}`}
        style={{
          backgroundColor: `rgb(${intensity}, ${intensity}, ${intensity})`
        }}
      />
    );
  };

  // Render overlay pixel (interactive layer)
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
          ? `Pixel (${x}, ${y}) revealed by ${revealedBy} - Value: ${pixelValue?.toFixed(3) || 'N/A'}`
          : canClick
            ? `Pixel (${x}, ${y}) - Value: ${pixelValue?.toFixed(3) || 'N/A'} - Click to reveal`
            : `Pixel (${x}, ${y}) - Value: ${pixelValue?.toFixed(3) || 'N/A'} - Not available (below threshold)`
        }
      />
    );
  };

  // Generate background image pixels
  const backgroundPixels = [];
  for (let y = 0; y < GAME_CONFIG.GRID_SIZE; y++) {
    for (let x = 0; x < GAME_CONFIG.GRID_SIZE; x++) {
      backgroundPixels.push(renderBackgroundPixel(x, y));
    }
  }

  // Generate overlay (interactive) pixels
  const overlayPixels = [];
  for (let y = 0; y < GAME_CONFIG.GRID_SIZE; y++) {
    for (let x = 0; x < GAME_CONFIG.GRID_SIZE; x++) {
      overlayPixels.push(renderOverlayPixel(x, y));
    }
  }

  // Create a small preview of the full image
  const PreviewImage = styled.div`
    width: 84px;
    height: 84px;
    display: grid;
    grid-template-columns: repeat(28, 1fr);
    grid-template-rows: repeat(28, 1fr);
    border: 1px solid #ccc;
    margin: 0 10px;
    opacity: 0.6;
    
    .preview-pixel {
      width: 100%;
      height: 100%;
    }
  `;

  const renderPreviewPixel = (x, y) => {
    const pixelValue = imageData[y][x];
    const normalizedValue = typeof pixelValue === 'number' ? Math.max(0, Math.min(1, pixelValue)) : 0;
    
    // Apply same contrast stretching as main image for consistency
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
        style={{
          backgroundColor: `rgb(${intensity}, ${intensity}, ${intensity})`
        }}
      />
    );
  };

  const previewPixels = [];
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      previewPixels.push(renderPreviewPixel(x, y));
    }
  }

  return (
    <BoardContainer>
      {/* Turn indicator with image preview */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
        <TurnIndicator isPlayerTurn={isPlayerTurn}>
          {loading 
            ? 'Processing move...'
            : isPlayerTurn 
              ? 'Your Turn - Click a pixel to reveal'
              : 'AI is thinking...'
          }
        </TurnIndicator>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ fontSize: '12px', marginBottom: '4px', opacity: 0.7 }}>
            Full Image (Reference)
          </div>
          <PreviewImage>
            {previewPixels}
          </PreviewImage>
        </div>
      </div>

      {/* Game grid with background image and overlay */}
      <GridContainer>
        {/* Background: Full MNIST image always visible */}
        <BackgroundImage>
          {backgroundPixels}
        </BackgroundImage>
        
        {/* Overlay: Interactive layer for game mechanics */}
        <GridOverlay>
          {overlayPixels}
        </GridOverlay>
      </GridContainer>

      {/* Game statistics */}
      <GameStats>
        <StatItem>
          <div className="label">Turn</div>
          <div className="value">{current_turn} / {config.k}</div>
        </StatItem>
        
        <StatItem>
          <div className="label">Pixels Revealed</div>
          <div className="value">{current_turn}</div>
        </StatItem>
        
        <StatItem>
          <div className="label">Remaining</div>
          <div className="value">{config.k - current_turn}</div>
        </StatItem>
        
        {judge_prediction && (
          <StatItem>
            <div className="label">Judge Prediction</div>
            <div className="value">{judge_prediction.predicted_class}</div>
          </StatItem>
        )}
        
        {judge_prediction && (
          <StatItem>
            <div className="label">Confidence</div>
            <div className="value">{(judge_prediction.confidence * 100).toFixed(1)}%</div>
          </StatItem>
        )}
      </GameStats>

      {/* Move legend */}
      <div style={{ 
        display: 'flex', 
        gap: '20px', 
        justifyContent: 'center',
        fontSize: '14px',
        color: '#666'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '20px', 
            height: '20px', 
            backgroundColor: '#e0e0e0',
            border: '1px solid #ccc' 
          }}></div>
          <span>Unrevealed</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '20px', 
            height: '20px', 
            backgroundColor: '#fff',
            border: '2px solid #4caf50' 
          }}></div>
          <span>Your moves</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '20px', 
            height: '20px', 
            backgroundColor: '#fff',
            border: '2px solid #f44336' 
          }}></div>
          <span>AI moves</span>
        </div>
      </div>
    </BoardContainer>
  );
};

export default GameBoard;