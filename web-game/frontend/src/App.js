import React, { useEffect } from 'react';
import styled from 'styled-components';
import { useGame } from './hooks/useGame';
import GameBoard from './components/GameBoard';
import GameConfig from './components/GameConfig';
import GameResult from './components/GameResult';
import ErrorBoundary from './components/ErrorBoundary';
import DebugInfo from './components/DebugInfo';
import { COLORS } from './utils/constants';
import { isGameOver } from './utils/helpers';

const AppContainer = styled.div`
  min-height: 100vh;
  background: ${COLORS.background};
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 32px;
  color: white;
  
  h1 {
    font-size: 2.5rem;
    margin: 0 0 12px 0;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
  }
  
  .subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 300;
    margin: 0;
  }
  
  .description {
    font-size: 1rem;
    opacity: 0.8;
    margin-top: 8px;
    max-width: 600px;
  }
`;

const MainContent = styled.main`
  display: flex;
  gap: 32px;
  max-width: 1200px;
  width: 100%;
  align-items: flex-start;
  flex-wrap: wrap;
  justify-content: center;
  
  @media (max-width: 1024px) {
    flex-direction: column;
    align-items: center;
  }
`;

const GameSection = styled.div`
  flex: 1;
  min-width: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
`;

const ConfigSection = styled.div`
  flex: 0 0 400px;
  
  @media (max-width: 1024px) {
    flex: none;
    width: 100%;
    max-width: 400px;
  }
`;

const ErrorBanner = styled.div`
  background: #ffebee;
  color: #c62828;
  padding: 16px 24px;
  border-radius: 8px;
  border-left: 4px solid #c62828;
  margin-bottom: 20px;
  max-width: 800px;
  width: 100%;
  text-align: center;
  font-weight: 500;
  
  button {
    background: none;
    border: none;
    color: #c62828;
    cursor: pointer;
    font-weight: 600;
    text-decoration: underline;
    margin-left: 8px;
    
    &:hover {
      color: #d32f2f;
    }
  }
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease-in-out infinite;
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const StatusIndicator = styled.div`
  background: rgba(255, 255, 255, 0.1);
  color: white;
  padding: 12px 24px;
  border-radius: 20px;
  margin-bottom: 20px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 12px;
`;

function AppContent() {
  const {
    // Game state
    gameState,
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
    resetGame,
    updateConfig,
    handlePixelHover,
    handlePixelClick,
    clearError,
  } = useGame();

  // Auto-clear errors after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(clearError, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, clearError]);

  const handleStartGame = () => {
    createGame(gameConfig);
  };

  const handleNewGame = () => {
    resetGame();
  };

  const gameIsOver = gameState && isGameOver(gameState);

  return (
    <AppContainer>
      <Header>
        <h1>AI Pixel Debate</h1>
        <p className="subtitle">Challenge AI in Strategic Pixel Revelation</p>
        <p className="description">
          Work together or against an AI agent to convince a judge classifier 
          by strategically revealing pixels from MNIST digit images.
        </p>
      </Header>

      {/* Error Banner */}
      {error && (
        <ErrorBanner>
          {error}
          <button onClick={clearError}>Dismiss</button>
        </ErrorBanner>
      )}

      {/* Status Indicator */}
      {loading && (
        <StatusIndicator>
          <LoadingSpinner />
          {gameState ? 'Processing move...' : 'Creating game...'}
        </StatusIndicator>
      )}

      <MainContent>
        {/* Game Board Section */}
        <GameSection>
          {isGameActive && !gameIsOver && (
            <GameBoard
              gameState={gameState}
              imageData={imageData}
              hoveredPixel={hoveredPixel}
              selectedPixel={selectedPixel}
              isPlayerTurn={isPlayerTurn}
              onPixelClick={handlePixelClick}
              onPixelHover={handlePixelHover}
              loading={loading}
            />
          )}

          {/* Game Result */}
          {gameIsOver && gameResult && (
            <GameResult
              gameResult={gameResult}
              gameState={gameState}
              onNewGame={handleNewGame}
            />
          )}

          {/* Welcome Message */}
          {!gameState && (
            <div style={{
              background: COLORS.cardBackground,
              padding: '40px',
              borderRadius: '12px',
              textAlign: 'center',
              maxWidth: '600px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
            }}>
              <h2 style={{ color: '#333', marginBottom: '16px' }}>
                Welcome to AI Pixel Debate!
              </h2>
              <p style={{ color: '#666', lineHeight: '1.6', marginBottom: '24px' }}>
                This game implements "AI Safety via Debate" on MNIST digit classification. 
                You and an AI agent will take turns revealing pixels to convince a judge 
                classifier about the correct digit.
              </p>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '16px',
                marginBottom: '24px',
                fontSize: '14px'
              }}>
                <div style={{ padding: '16px', background: '#f8f9fa', borderRadius: '8px' }}>
                  <strong style={{ color: COLORS.success }}>As Honest Player</strong>
                  <p style={{ margin: '8px 0 0 0', color: '#666' }}>
                    Help the judge classify the digit correctly
                  </p>
                </div>
                <div style={{ padding: '16px', background: '#f8f9fa', borderRadius: '8px' }}>
                  <strong style={{ color: COLORS.error }}>As Liar Player</strong>
                  <p style={{ margin: '8px 0 0 0', color: '#666' }}>
                    Try to mislead the judge into wrong classification
                  </p>
                </div>
              </div>
              <p style={{ color: '#888', fontSize: '12px' }}>
                Configure your game settings on the right and click "Start Game" to begin!
              </p>
            </div>
          )}
        </GameSection>

        {/* Configuration Section */}
        <ConfigSection>
          <GameConfig
            config={gameConfig}
            onConfigChange={updateConfig}
            onStartGame={handleStartGame}
            onResetGame={handleNewGame}
            loading={loading}
            error={null} // We show errors in the main banner
            gameActive={isGameActive}
          />

          {/* Game Stats */}
          {isGameActive && (
            <div style={{
              background: COLORS.cardBackground,
              borderRadius: '12px',
              padding: '20px',
              marginTop: '20px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
            }}>
              <h4 style={{ margin: '0 0 16px 0', color: '#333' }}>
                Current Game
              </h4>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                fontSize: '14px'
              }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  borderBottom: '1px solid #eee'
                }}>
                  <span style={{ color: '#666' }}>Progress</span>
                  <strong>{currentTurn} / {totalPixels}</strong>
                </div>
                
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  borderBottom: '1px solid #eee'
                }}>
                  <span style={{ color: '#666' }}>Remaining</span>
                  <strong>{pixelsRemaining}</strong>
                </div>
                
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  borderBottom: '1px solid #eee'
                }}>
                  <span style={{ color: '#666' }}>Your Role</span>
                  <strong>{gameConfig.player_role}</strong>
                </div>
                
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  borderBottom: '1px solid #eee'
                }}>
                  <span style={{ color: '#666' }}>AI Type</span>
                  <strong>{gameConfig.agent_type}</strong>
                </div>

                {gameState?.judge_prediction && (
                  <>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '8px 0',
                      borderBottom: '1px solid #eee',
                      gridColumn: '1 / -1'
                    }}>
                      <span style={{ color: '#666' }}>Judge Thinks</span>
                      <strong>Digit {gameState.judge_prediction.predicted_class}</strong>
                    </div>
                    
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '8px 0',
                      gridColumn: '1 / -1'
                    }}>
                      <span style={{ color: '#666' }}>Confidence</span>
                      <strong>
                        {(gameState.judge_prediction.confidence * 100).toFixed(1)}%
                      </strong>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </ConfigSection>
      </MainContent>
    </AppContainer>
  );
}

function App() {
  // Add some initial logging
  console.log('App component mounting...');
  
  return (
    <ErrorBoundary>
      <AppContent />
      <DebugInfo />
    </ErrorBoundary>
  );
}

export default App;