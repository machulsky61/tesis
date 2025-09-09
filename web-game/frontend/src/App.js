import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';
import { useGame } from './hooks/useGame';
import GameBoard from './components/GameBoard';
import GameResult from './components/GameResult';
import ErrorBoundary from './components/ErrorBoundary';
import DebugInfo from './components/DebugInfo';
import Navbar from './components/ui/Navbar';
import CollapsibleSidebar from './components/ui/CollapsibleSidebar';
import LandingPage from './components/LandingPage';
import { Card, GlassCard } from './components/ui/Card';
import ThemeProvider from './theme/ThemeProvider';
import { GlobalStyles } from './theme/GlobalStyles';
import { isGameOver } from './utils/helpers';

const AppContainer = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.background.primary};
  position: relative;
  overflow-x: hidden;
`;

const BackgroundPattern = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  opacity: 0.03;
  background-image: 
    radial-gradient(circle at 25% 25%, ${props => props.theme.brand.primary} 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, ${props => props.theme.brand.secondary} 0%, transparent 50%);
  pointer-events: none;
`;

const MainContent = styled(motion.main)`
  padding: 2rem;
  padding-top: 6rem; /* Account for fixed navbar */
  padding-left: 4rem; /* Account for collapsed sidebar */
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
  min-height: 100vh;
  
  @media (max-width: 768px) {
    padding: 1rem;
    padding-top: 5rem; /* Smaller top padding on mobile */
    padding-left: 1rem; /* Normal padding on mobile */
    gap: 1.5rem;
  }
`;

const GameSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  align-items: center;
  width: 100%;
  max-width: 800px;
`;

const WelcomeCard = styled(Card)`
  text-align: center;
  max-width: 600px;
  margin: 2rem auto;

  h2 {
    color: ${props => props.theme.text.primary};
    margin-bottom: 1rem;
    font-size: 1.75rem;
  }

  .description {
    color: ${props => props.theme.text.secondary};
    line-height: 1.6;
    margin-bottom: 2rem;
  }

  .roles-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;

    @media (max-width: 640px) {
      grid-template-columns: 1fr;
    }
  }

  .role-card {
    padding: 1rem;
    background: ${props => props.theme.background.glass};
    border-radius: ${props => props.theme.radius.md};
    border: 1px solid ${props => props.theme.background.glass};

    .role-title {
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    .role-desc {
      font-size: 0.875rem;
      color: ${props => props.theme.text.secondary};
      margin: 0;
    }

    &.honest .role-title {
      color: ${props => props.theme.semantic.honest};
    }

    &.liar .role-title {
      color: ${props => props.theme.semantic.liar};
    }
  }

  .start-prompt {
    color: ${props => props.theme.text.muted};
    font-size: 0.875rem;
  }
`;

const StatusCard = styled(GlassCard)`
  text-align: center;
  padding: 1rem 1.5rem;
  margin-bottom: 1rem;

  .status-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
  }

  .spinner {
    width: 1.25rem;
    height: 1.25rem;
    border: 2px solid ${props => props.theme.interactive.disabled};
    border-radius: 50%;
    border-top-color: ${props => props.theme.brand.primary};
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .status-text {
    font-weight: 500;
    color: ${props => props.theme.text.primary};
  }
`;

function AppContent() {
  const [gameStats] = useState({ games: 0, accuracy: 0, streak: 0 });
  const [showLanding, setShowLanding] = useState(true);
  
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
    
    // Actions
    createGame,
    resetGame,
    updateConfig,
    handlePixelHover,
    handlePixelClick,
    clearError,
  } = useGame();

  // Handle errors with toast notifications
  useEffect(() => {
    if (error) {
      toast.error(error, {
        duration: 5000,
        style: {
          background: 'var(--bg-surface)',
          color: 'var(--text-primary)',
        },
      });
      clearError();
    }
  }, [error, clearError]);

  // Portfolio integration handlers
  const handleAbout = () => {
    // TODO: Implement about modal
    toast('About section coming soon!', { icon: 'ℹ️' });
  };

  const handleGithub = () => {
    window.open('https://github.com/machulsky61/tesis', '_blank');
  };

  const handleThesis = () => {
    // TODO: Link to thesis PDF
    toast('Thesis link coming soon!', { icon: '📄' });
  };

  const handleStartGame = () => {
    setShowLanding(false);
    createGame(gameConfig);
  };

  const handleNewGame = () => {
    resetGame();
    setShowLanding(true);
  };

  const gameIsOver = gameState && isGameOver(gameState);

  return (
    <AppContainer>
      <BackgroundPattern />
      
      <AnimatePresence mode="wait">
        {showLanding ? (
          <LandingPage
            key="landing"
            onStartGame={handleStartGame}
            onAbout={handleAbout}
            onGithub={handleGithub}
            onThesis={handleThesis}
          />
        ) : (
          <motion.div
            key="game-view"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            {/* Game Navbar - consistent with landing page */}
            <Navbar onStartGame={() => setShowLanding(true)} />
            
            {/* Collapsible Sidebar */}
            <CollapsibleSidebar
              config={gameConfig}
              onConfigChange={updateConfig}
              onStartGame={handleStartGame}
              onResetGame={handleNewGame}
              loading={loading}
              error={null}
              gameActive={isGameActive}
            />

            <MainContent
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <GameSection>
                <AnimatePresence mode="wait">
                  {/* Show loading only when creating game (no gameState yet) */}
                  {loading && !gameState && (
                    <StatusCard
                      key="loading"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                    >
                      <div className="status-content">
                        <div className="spinner" />
                        <span className="status-text">Creating game...</span>
                      </div>
                    </StatusCard>
                  )}

                  {/* Game Board - Always visible when game exists (handles its own loading overlay) */}
                  {isGameActive && !gameIsOver && (
                    <motion.div
                      key="game"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ duration: 0.3 }}
                    >
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
                    </motion.div>
                  )}

                  {/* Game Result */}
                  {gameIsOver && gameResult && (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ duration: 0.3 }}
                    >
                      <GameResult
                        gameResult={gameResult}
                        gameState={gameState}
                        onNewGame={handleNewGame}
                      />
                    </motion.div>
                  )}

                  {/* Welcome Screen */}
                  {!gameState && !loading && (
                    <WelcomeCard
                      key="welcome"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.5 }}
                    >
                      <h2>Welcome to AI Pixel Debate!</h2>
                      <p className="description">
                        This interactive game implements "AI Safety via Debate" methodology 
                        on MNIST digit classification. Challenge AI agents by strategically 
                        revealing pixels to convince a neural judge.
                      </p>
                      
                      <div className="roles-grid">
                        <div className="role-card honest">
                          <div className="role-title">Honest Player</div>
                          <p className="role-desc">
                            Help the judge classify the digit correctly by revealing 
                            informative pixels
                          </p>
                        </div>
                        <div className="role-card liar">
                          <div className="role-title">Liar Player</div>
                          <p className="role-desc">
                            Try to mislead the judge into making wrong classifications 
                            through strategic deception
                          </p>
                        </div>
                      </div>
                      
                      <p className="start-prompt">
                        Configure your game settings and click "Start Game" to begin the debate!
                      </p>
                    </WelcomeCard>
                  )}
                </AnimatePresence>
              </GameSection>
            </MainContent>
          </motion.div>
        )}
      </AnimatePresence>
    </AppContainer>
  );
}

function App() {
  console.log('App component mounting with modern UI...');
  
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <GlobalStyles />
        <AppContent />
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'var(--bg-surface)',
              color: 'var(--text-primary)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '0.75rem',
              backdropFilter: 'blur(8px)',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#ffffff',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#ffffff',
              },
            },
          }}
        />
        <DebugInfo />
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;