import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Settings, 
  Target, 
  User, 
  Bot, 
  Zap, 
  Users, 
  Cpu,
  ChevronLeft,
  ChevronRight,
  Play,
  RotateCcw
} from 'lucide-react';
import { PLAYER_ROLES, AGENT_TYPES, GAME_CONFIG } from '../../utils/constants';
import { getRoleDescription, getAgentDescription } from '../../utils/helpers';
import { Button, SecondaryButton } from './Button';

const SidebarContainer = styled(motion.div)`
  position: fixed;
  left: 0;
  top: 5rem; /* Start below navbar */
  bottom: 0;
  z-index: 50; /* Lower than navbar */
  display: flex;
  
  @media (max-width: 768px) {
    position: fixed;
    top: 4rem; /* Smaller top offset on mobile */
    bottom: auto;
    height: calc(100vh - 4rem);
  }
`;

const SidebarTrigger = styled(motion.div)`
  width: ${props => props.isExpanded ? '0px' : '60px'};
  background: ${props => props.theme.background.glass};
  backdrop-filter: ${props => props.theme.blur.md};
  border: 1px solid ${props => props.theme.background.glass};
  border-left: none;
  border-top-right-radius: ${props => props.theme.radius.lg};
  border-bottom-right-radius: ${props => props.theme.radius.lg};
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem 0;
  gap: 0.75rem;
  overflow: hidden;
  transition: all 400ms cubic-bezier(0.175, 0.885, 0.32, 1.275);
  
  .trigger-icon {
    width: 1.5rem;
    height: 1.5rem;
    color: ${props => props.theme.brand.primary};
    cursor: pointer;
    padding: 0.5rem;
    border-radius: ${props => props.theme.radius.md};
    transition: all 200ms ease;
    
    &:hover {
      background: ${props => props.theme.interactive.hover};
      transform: scale(1.1);
    }
  }
  
  .section-icon {
    width: 1.25rem;
    height: 1.25rem;
    color: ${props => props.theme.text.secondary};
    opacity: 0.7;
    transition: all 200ms ease;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: ${props => props.theme.radius.sm};
    
    &:hover {
      opacity: 1;
      color: ${props => props.theme.brand.primary};
      background: ${props => props.theme.interactive.hover};
    }
  }
`;

const SidebarContent = styled(motion.div)`
  width: 320px;
  background: ${props => props.theme.background.surface};
  backdrop-filter: ${props => props.theme.blur.lg};
  border: 1px solid ${props => props.theme.background.glass};
  border-left: none;
  border-top-right-radius: ${props => props.theme.radius.lg};
  border-bottom-right-radius: ${props => props.theme.radius.lg};
  box-shadow: ${props => props.theme.shadows.xl};
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const SidebarHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid ${props => props.theme.background.glass};
  
  .header-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.125rem;
    font-weight: 700;
    color: ${props => props.theme.text.primary};
    
    .header-icon {
      width: 1.25rem;
      height: 1.25rem;
      color: ${props => props.theme.brand.primary};
    }
  }
  
  .close-btn {
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: ${props => props.theme.radius.sm};
    cursor: pointer;
    color: ${props => props.theme.text.secondary};
    transition: all 200ms ease;
    
    &:hover {
      background: ${props => props.theme.interactive.hover};
      color: ${props => props.theme.text.primary};
    }
  }
`;

const SidebarBody = styled.div`
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const SidebarSection = styled(motion.div)`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const SectionLabel = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  color: ${props => props.theme.text.primary};
  
  .section-icon {
    width: 1rem;
    height: 1rem;
    color: ${props => props.theme.brand.primary};
  }
`;

const SectionDescription = styled.div`
  font-size: 0.75rem;
  color: ${props => props.theme.text.secondary};
  line-height: 1.4;
  margin-bottom: 0.5rem;
`;

const StyledSelect = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 2px solid ${props => props.theme.background.glass};
  border-radius: ${props => props.theme.radius.md};
  font-size: 0.875rem;
  background: ${props => props.theme.background.surface};
  color: ${props => props.theme.text.primary};
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);
  cursor: pointer;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.brand.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.brand.primary}20;
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  option {
    background: ${props => props.theme.background.surface};
    color: ${props => props.theme.text.primary};
  }
`;

const RangeContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  background: ${props => props.theme.background.glass};
  border-radius: ${props => props.theme.radius.md};
  border: 1px solid ${props => props.theme.background.glass};
`;

const RangeInput = styled.input`
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: ${props => props.theme.interactive.disabled};
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    appearance: none;
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 50%;
    background: ${props => props.theme.brand.primary};
    cursor: pointer;
    box-shadow: ${props => props.theme.shadows.md};
    transition: all 200ms ease;
    
    &:hover {
      transform: scale(1.1);
    }
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const RangeValue = styled(motion.div)`
  font-weight: 700;
  font-size: 1rem;
  color: ${props => props.theme.brand.primary};
  min-width: 2rem;
  text-align: center;
  padding: 0.25rem 0.5rem;
  background: ${props => props.theme.brand.primary}15;
  border-radius: ${props => props.theme.radius.sm};
`;

const ToggleContainer = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: ${props => props.theme.background.glass};
  border-radius: ${props => props.theme.radius.md};
  border: 1px solid ${props => props.theme.background.glass};
  cursor: pointer;
  transition: all 300ms ease;

  &:hover:not(.disabled) {
    border-color: ${props => props.theme.brand.primary}40;
    background: ${props => props.theme.brand.primary}05;
  }

  &.disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const Toggle = styled.div`
  position: relative;
  width: 2.25rem;
  height: 1.125rem;
  background: ${props => props.checked ? props.theme.brand.primary : props.theme.interactive.disabled};
  border-radius: ${props => props.theme.radius.full};
  transition: all 300ms ease;

  &::before {
    content: '';
    position: absolute;
    top: 0.125rem;
    left: ${props => props.checked ? '1.125rem' : '0.125rem'};
    width: 0.875rem;
    height: 0.875rem;
    background: ${props => props.theme.text.inverse};
    border-radius: 50%;
    box-shadow: ${props => props.theme.shadows.sm};
    transition: all 300ms ease;
  }
`;

const ToggleLabel = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.text.primary};
  font-weight: 500;
  flex: 1;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  border-top: 1px solid ${props => props.theme.background.glass};
`;

const CollapsibleSidebar = ({
  config,
  onConfigChange,
  onStartGame,
  onResetGame,
  loading,
  error,
  gameActive
}) => {
  const [isExpanded, setIsExpanded] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('sidebar-expanded') || 'true');
    } catch {
      return true;
    }
  });

  useEffect(() => {
    localStorage.setItem('sidebar-expanded', JSON.stringify(isExpanded));
  }, [isExpanded]);

  const handleChange = (field, value) => {
    onConfigChange({ [field]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onStartGame();
  };

  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  const sectionVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  };

  return (
    <SidebarContainer>
      {/* Trigger */}
      <SidebarTrigger isExpanded={isExpanded}>
        <motion.div
          className="trigger-icon"
          onClick={toggleSidebar}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          {isExpanded ? <ChevronLeft /> : <Settings />}
        </motion.div>
        
        {!isExpanded && (
          <>
            <Target className="section-icon" title="Pixels" />
            <User className="section-icon" title="Role" />
            <Bot className="section-icon" title="AI" />
            <Zap className="section-icon" title="Strategy" />
            <Users className="section-icon" title="Judge" />
            <Cpu className="section-icon" title="Mode" />
          </>
        )}
      </SidebarTrigger>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <SidebarContent
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ 
              type: "spring", 
              stiffness: 300, 
              damping: 30 
            }}
          >
            <SidebarHeader>
              <div className="header-title">
                <Settings className="header-icon" />
                {gameActive ? 'Game Settings' : 'Configuration'}
              </div>
              <div className="close-btn" onClick={toggleSidebar}>
                <ChevronLeft />
              </div>
            </SidebarHeader>

            <SidebarBody>
              <form onSubmit={handleSubmit}>
                {/* Pixels Section */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.1 }}
                >
                  <SectionLabel>
                    <Target className="section-icon" />
                    Pixels to Reveal
                  </SectionLabel>
                  <SectionDescription>
                    Total pixels both players will reveal. Higher = easier classification.
                  </SectionDescription>
                  <RangeContainer>
                    <RangeInput
                      type="range"
                      min={GAME_CONFIG.MIN_PIXELS}
                      max={GAME_CONFIG.MAX_PIXELS}
                      value={config.k}
                      onChange={(e) => handleChange('k', parseInt(e.target.value))}
                      disabled={gameActive}
                    />
                    <RangeValue
                      key={config.k}
                      initial={{ scale: 1.2 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring" }}
                    >
                      {config.k}
                    </RangeValue>
                  </RangeContainer>
                </SidebarSection>

                {/* Player Role */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.2 }}
                >
                  <SectionLabel>
                    <User className="section-icon" />
                    Your Role
                  </SectionLabel>
                  <SectionDescription>
                    {getRoleDescription(config.player_role)}
                  </SectionDescription>
                  <StyledSelect
                    value={config.player_role}
                    onChange={(e) => handleChange('player_role', e.target.value)}
                    disabled={gameActive}
                  >
                    <option value={PLAYER_ROLES.HONEST}>
                      🎯 Honest Player
                    </option>
                    <option value={PLAYER_ROLES.LIAR}>
                      🎭 Liar Player
                    </option>
                  </StyledSelect>
                </SidebarSection>

                {/* AI Opponent */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.3 }}
                >
                  <SectionLabel>
                    <Bot className="section-icon" />
                    AI Opponent
                  </SectionLabel>
                  <SectionDescription>
                    {getAgentDescription(config.agent_type)}
                  </SectionDescription>
                  <StyledSelect
                    value={config.agent_type}
                    onChange={(e) => handleChange('agent_type', e.target.value)}
                    disabled={gameActive}
                  >
                    <option value={AGENT_TYPES.GREEDY}>
                      ⚡ Greedy AI
                    </option>
                    <option value={AGENT_TYPES.MCTS}>
                      🧠 Strategic AI
                    </option>
                  </StyledSelect>
                </SidebarSection>

                {/* Precommit Strategy */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.4 }}
                >
                  <SectionLabel>
                    <Zap className="section-icon" />
                    Precommit Strategy
                  </SectionLabel>
                  <SectionDescription>
                    Liar commits to wrong answer upfront.
                  </SectionDescription>
                  <ToggleContainer
                    className={gameActive ? 'disabled' : ''}
                    onClick={() => !gameActive && handleChange('precommit', !config.precommit)}
                    whileTap={!gameActive ? { scale: 0.98 } : {}}
                  >
                    <Toggle checked={config.precommit} />
                    <ToggleLabel>Enable precommit</ToggleLabel>
                  </ToggleContainer>
                </SidebarSection>

                {/* Judge Model */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.5 }}
                >
                  <SectionLabel>
                    <Users className="section-icon" />
                    Judge Model
                  </SectionLabel>
                  <SectionDescription>
                    AI judge for final classification.
                  </SectionDescription>
                  <StyledSelect
                    value={config.judge_name}
                    onChange={(e) => handleChange('judge_name', e.target.value)}
                    disabled={gameActive}
                  >
                    <option value="28">🏛️ Standard Judge</option>
                    <option value="16">📱 Compact Judge</option>
                  </StyledSelect>
                </SidebarSection>

                {/* Pixel Mode */}
                <SidebarSection
                  variants={sectionVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.6 }}
                >
                  <SectionLabel>
                    <Cpu className="section-icon" />
                    Pixel Selection
                  </SectionLabel>
                  <SectionDescription>
                    Allow black pixels (OOD mode).
                  </SectionDescription>
                  <ToggleContainer
                    className={gameActive ? 'disabled' : ''}
                    onClick={() => !gameActive && handleChange('allow_zero_pixels', !config.allow_zero_pixels)}
                    whileTap={!gameActive ? { scale: 0.98 } : {}}
                  >
                    <Toggle checked={config.allow_zero_pixels} />
                    <ToggleLabel>OOD Mode</ToggleLabel>
                  </ToggleContainer>
                </SidebarSection>
              </form>
            </SidebarBody>

            {/* Action Buttons */}
            <ActionButtons>
              <AnimatePresence mode="wait">
                {gameActive ? (
                  <SecondaryButton
                    key="reset"
                    type="button"
                    onClick={onResetGame}
                    disabled={loading}
                    style={{ flex: 1 }}
                  >
                    <RotateCcw style={{ width: '1rem', height: '1rem' }} />
                    New Game
                  </SecondaryButton>
                ) : (
                  <Button
                    key="start"
                    onClick={handleSubmit}
                    disabled={loading}
                    style={{ flex: 1 }}
                  >
                    <Play style={{ width: '1rem', height: '1rem' }} />
                    {loading ? 'Creating...' : 'Start Game'}
                  </Button>
                )}
              </AnimatePresence>
            </ActionButtons>
          </SidebarContent>
        )}
      </AnimatePresence>
    </SidebarContainer>
  );
};

export default CollapsibleSidebar;