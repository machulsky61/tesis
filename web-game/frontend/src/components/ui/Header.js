import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Zap, Github, BookOpen, Info } from 'lucide-react';
import ThemeToggle from './ThemeToggle';
import { IconButton, GhostButton } from './Button';

const HeaderContainer = styled(motion.header)`
  position: sticky;
  top: 0;
  z-index: 100;
  width: 100%;
  background: ${props => props.theme.background.glass};
  backdrop-filter: ${props => props.theme.blur.md};
  border-bottom: 1px solid ${props => props.theme.background.glass};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const HeaderContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  max-width: 1200px;
  margin: 0 auto;

  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const LogoSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
`;

const Logo = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  
  svg {
    width: 2rem;
    height: 2rem;
    color: ${props => props.theme.brand.primary};
  }
`;

const LogoText = styled.div`
  display: flex;
  flex-direction: column;
  
  h1 {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
    background: ${props => props.theme.brand.gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .subtitle {
    font-size: 0.75rem;
    font-weight: 500;
    color: ${props => props.theme.text.secondary};
    margin: 0;
    opacity: 0.8;
  }

  @media (max-width: 640px) {
    .subtitle {
      display: none;
    }
  }
`;

const NavSection = styled.nav`
  display: flex;
  align-items: center;
  gap: 0.5rem;

  @media (max-width: 640px) {
    gap: 0.25rem;
  }
`;

const NavButton = styled(GhostButton)`
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  border-radius: ${props => props.theme.radius.md};

  svg {
    width: 1rem;
    height: 1rem;
  }

  @media (max-width: 768px) {
    padding: 0.5rem;
    min-width: 0;
    
    span {
      display: none;
    }
  }
`;

const StatsSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
  justify-content: center;

  @media (max-width: 768px) {
    display: none;
  }
`;

const StatItem = styled.div`
  text-align: center;
  
  .value {
    font-size: 1.125rem;
    font-weight: 700;
    color: ${props => props.theme.brand.primary};
    line-height: 1;
  }
  
  .label {
    font-size: 0.75rem;
    color: ${props => props.theme.text.secondary};
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
`;

const Header = ({ 
  gameStats = null, 
  onAbout = null, 
  onGithub = null, 
  onThesis = null,
  showStats = true 
}) => {
  const handleLogoClick = () => {
    window.location.reload();
  };

  const defaultStats = {
    games: 0,
    accuracy: 0,
    streak: 0
  };

  const stats = gameStats || defaultStats;

  return (
    <HeaderContainer
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <HeaderContent>
        <LogoSection>
          <Logo
            onClick={handleLogoClick}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Zap />
            <LogoText>
              <h1>AI Pixel Debate</h1>
              <p className="subtitle">Strategic Pixel Revelation</p>
            </LogoText>
          </Logo>
        </LogoSection>

        {showStats && (
          <StatsSection>
            <StatItem>
              <div className="value">{stats.games}</div>
              <div className="label">Games</div>
            </StatItem>
            <StatItem>
              <div className="value">{stats.accuracy}%</div>
              <div className="label">Accuracy</div>
            </StatItem>
            <StatItem>
              <div className="value">{stats.streak}</div>
              <div className="label">Streak</div>
            </StatItem>
          </StatsSection>
        )}

        <NavSection>
          {onAbout && (
            <NavButton onClick={onAbout}>
              <Info />
              <span>About</span>
            </NavButton>
          )}
          
          {onThesis && (
            <NavButton onClick={onThesis}>
              <BookOpen />
              <span>Thesis</span>
            </NavButton>
          )}
          
          {onGithub && (
            <NavButton onClick={onGithub}>
              <Github />
              <span>Code</span>
            </NavButton>
          )}
          
          <ThemeToggle compact />
        </NavSection>
      </HeaderContent>
    </HeaderContainer>
  );
};

export default Header;