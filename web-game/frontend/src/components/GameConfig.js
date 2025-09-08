import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Bot, Zap, Settings, Play, RotateCcw, HelpCircle, Target, Users, Cpu } from 'lucide-react';
import { PLAYER_ROLES, AGENT_TYPES, GAME_CONFIG } from '../utils/constants';
import { getRoleDescription, getAgentDescription } from '../utils/helpers';
import { Card } from './ui/Card';
import { Button, SecondaryButton } from './ui/Button';

const ConfigContainer = styled(Card)`
  max-width: 400px;
  width: 100%;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const ConfigTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  
  h3 {
    margin: 0;
    color: ${props => props.theme.text.primary};
    font-size: 1.25rem;
    font-weight: 700;
  }
  
  .config-icon {
    width: 1.5rem;
    height: 1.5rem;
    color: ${props => props.theme.brand.primary};
  }
`;

const ConfigSection = styled(motion.div)`
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

const Description = styled.div`
  font-size: 0.75rem;
  color: ${props => props.theme.text.secondary};
  line-height: 1.5;
  background: ${props => props.theme.background.glass};
  padding: 0.75rem;
  border-radius: ${props => props.theme.radius.md};
  border-left: 3px solid ${props => props.theme.brand.primary};
  backdrop-filter: ${props => props.theme.blur.sm};
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

  &:hover:not(:disabled) {
    border-color: ${props => props.theme.brand.primary}60;
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
  transition: all 200ms ease;
  
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
      box-shadow: ${props => props.theme.shadows.lg};
    }
  }
  
  &::-moz-range-thumb {
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 50%;
    background: ${props => props.theme.brand.primary};
    cursor: pointer;
    border: none;
    box-shadow: ${props => props.theme.shadows.md};
    transition: all 200ms ease;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    
    &::-webkit-slider-thumb {
      cursor: not-allowed;
      transform: none;
    }
  }
`;

const RangeValue = styled(motion.div)`
  font-weight: 700;
  font-size: 1.125rem;
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
  padding: 1rem;
  background: ${props => props.theme.background.glass};
  border-radius: ${props => props.theme.radius.md};
  border: 1px solid ${props => props.theme.background.glass};
  cursor: pointer;
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);

  &:hover {
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
  width: 2.5rem;
  height: 1.25rem;
  background: ${props => props.checked ? props.theme.brand.primary : props.theme.interactive.disabled};
  border-radius: ${props => props.theme.radius.full};
  cursor: pointer;
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);

  &::before {
    content: '';
    position: absolute;
    top: 0.125rem;
    left: ${props => props.checked ? '1.375rem' : '0.125rem'};
    width: 1rem;
    height: 1rem;
    background: ${props => props.theme.text.inverse};
    border-radius: 50%;
    box-shadow: ${props => props.theme.shadows.sm};
    transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }

  &:hover::before {
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const ToggleLabel = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.text.primary};
  cursor: pointer;
  flex: 1;
  font-weight: 500;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 0.75rem;
  margin-top: 0.5rem;

  @media (max-width: 640px) {
    flex-direction: column;
    
    button {
      width: 100%;
    }
  }
`;

const ErrorMessage = styled(motion.div)`
  background: ${props => props.theme.semantic.error}10;
  color: ${props => props.theme.semantic.error};
  padding: 0.75rem;
  border-radius: ${props => props.theme.radius.md};
  border-left: 4px solid ${props => props.theme.semantic.error};
  font-size: 0.875rem;
  font-weight: 500;
  backdrop-filter: ${props => props.theme.blur.sm};
`;

const GameStatsCard = styled(Card)`
  margin-top: 1rem;
  padding: 1rem;

  h4 {
    margin: 0 0 1rem 0;
    color: ${props => props.theme.text.primary};
    font-size: 1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;

    .stats-icon {
      width: 1rem;
      height: 1rem;
      color: ${props => props.theme.brand.primary};
    }
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    font-size: 0.875rem;

    @media (max-width: 480px) {
      grid-template-columns: 1fr;
    }
  }

  .stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid ${props => props.theme.background.glass};

    &:last-child {
      border-bottom: none;
    }

    .label {
      color: ${props => props.theme.text.secondary};
      font-weight: 500;
    }

    .value {
      font-weight: 600;
      color: ${props => props.theme.text.primary};
    }

    &.prediction {
      grid-column: 1 / -1;
      
      .value {
        color: ${props => props.theme.brand.primary};
      }
    }
  }
`;

const GameConfig = ({
  config,
  onConfigChange,
  onStartGame,
  onResetGame,
  loading,
  error,
  gameActive
}) => {
  const handleChange = (field, value) => {
    onConfigChange({ [field]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onStartGame();
  };

  const sectionVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <ConfigContainer
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <ConfigTitle>
        <Settings className="config-icon" />
        <h3>{gameActive ? 'Current Game' : 'Game Configuration'}</h3>
      </ConfigTitle>
      
      <form onSubmit={handleSubmit}>
        {/* Number of pixels */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <SectionLabel>
            <Target className="section-icon" />
            Pixels to Reveal
          </SectionLabel>
          <Description>
            Total pixels both players will reveal (you + AI). Higher values make classification easier.
          </Description>
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
              transition={{ type: "spring", stiffness: 300 }}
            >
              {config.k}
            </RangeValue>
          </RangeContainer>
        </ConfigSection>

        {/* Player role */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <SectionLabel>
            <User className="section-icon" />
            Your Role
          </SectionLabel>
          <Description>
            {getRoleDescription(config.player_role)}
          </Description>
          <StyledSelect
            value={config.player_role}
            onChange={(e) => handleChange('player_role', e.target.value)}
            disabled={gameActive}
          >
            <option value={PLAYER_ROLES.HONEST}>
              🎯 Honest Player (Help judge classify correctly)
            </option>
            <option value={PLAYER_ROLES.LIAR}>
              🎭 Liar Player (Try to mislead the judge)
            </option>
          </StyledSelect>
        </ConfigSection>

        {/* AI opponent type */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <SectionLabel>
            <Bot className="section-icon" />
            AI Opponent
          </SectionLabel>
          <Description>
            {getAgentDescription(config.agent_type)}
          </Description>
          <StyledSelect
            value={config.agent_type}
            onChange={(e) => handleChange('agent_type', e.target.value)}
            disabled={gameActive}
          >
            <option value={AGENT_TYPES.GREEDY}>
              ⚡ Greedy AI (Fast, immediate decisions)
            </option>
            <option value={AGENT_TYPES.MCTS}>
              🧠 Strategic AI (Slower, but plans ahead)
            </option>
          </StyledSelect>
        </ConfigSection>

        {/* Precommit strategy */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.4 }}
        >
          <SectionLabel>
            <Zap className="section-icon" />
            Precommit Strategy
          </SectionLabel>
          <Description>
            When enabled, the liar agent commits to a specific wrong answer at the start, 
            potentially making it more strategic but also more predictable.
          </Description>
          <ToggleContainer
            className={gameActive ? 'disabled' : ''}
            onClick={() => !gameActive && handleChange('precommit', !config.precommit)}
            whileTap={!gameActive ? { scale: 0.98 } : {}}
          >
            <Toggle checked={config.precommit} />
            <ToggleLabel>Enable precommit strategy</ToggleLabel>
          </ToggleContainer>
        </ConfigSection>

        {/* Judge model selection */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.5 }}
        >
          <SectionLabel>
            <Users className="section-icon" />
            Judge Model
          </SectionLabel>
          <Description>
            The AI judge that will evaluate the revealed pixels and make the final classification.
          </Description>
          <StyledSelect
            value={config.judge_name}
            onChange={(e) => handleChange('judge_name', e.target.value)}
            disabled={gameActive}
          >
            <option value="28">🏛️ Standard Judge (28x28 resolution)</option>
            <option value="16">📱 Compact Judge (16x16 resolution)</option>
          </StyledSelect>
        </ConfigSection>

        {/* Pixel selection threshold */}
        <ConfigSection
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: 0.6 }}
        >
          <SectionLabel>
            <Cpu className="section-icon" />
            Pixel Selection Mode
          </SectionLabel>
          <Description>
            Standard mode: Only bright pixels (intensity > 0). 
            OOD mode: Allow selecting black pixels too (intensity = 0).
          </Description>
          <ToggleContainer
            className={gameActive ? 'disabled' : ''}
            onClick={() => !gameActive && handleChange('allow_zero_pixels', !config.allow_zero_pixels)}
            whileTap={!gameActive ? { scale: 0.98 } : {}}
          >
            <Toggle checked={config.allow_zero_pixels} />
            <ToggleLabel>Enable OOD mode (allow black pixels)</ToggleLabel>
          </ToggleContainer>
        </ConfigSection>

        {/* Action buttons */}
        <ButtonGroup>
          <AnimatePresence mode="wait">
            {gameActive ? (
              <SecondaryButton
                key="reset"
                type="button"
                onClick={onResetGame}
                disabled={loading}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <RotateCcw style={{ width: '1rem', height: '1rem' }} />
                New Game
              </SecondaryButton>
            ) : (
              <Button
                key="start"
                type="submit"
                disabled={loading}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Play style={{ width: '1rem', height: '1rem' }} />
                {loading ? 'Creating Game...' : 'Start Game'}
              </Button>
            )}
          </AnimatePresence>
        </ButtonGroup>

        {/* Error display */}
        <AnimatePresence>
          {error && (
            <ErrorMessage
              initial={{ opacity: 0, y: -10, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -10, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {error}
            </ErrorMessage>
          )}
        </AnimatePresence>
      </form>
    </ConfigContainer>
  );
};

export default GameConfig;