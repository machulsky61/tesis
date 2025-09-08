import React from 'react';
import styled from 'styled-components';
import { COLORS, PLAYER_ROLES, AGENT_TYPES, GAME_CONFIG } from '../utils/constants';
import { getRoleDescription, getAgentDescription } from '../utils/helpers';

const ConfigContainer = styled.div`
  background: ${COLORS.cardBackground};
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  max-width: 400px;
  width: 100%;
`;

const ConfigTitle = styled.h3`
  margin: 0 0 20px 0;
  color: #333;
  font-size: 20px;
  font-weight: 600;
`;

const ConfigSection = styled.div`
  margin-bottom: 20px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #555;
  font-size: 14px;
`;

const Description = styled.div`
  font-size: 12px;
  color: #777;
  margin-bottom: 12px;
  line-height: 1.4;
  background: #f8f9fa;
  padding: 8px 12px;
  border-radius: 4px;
  border-left: 3px solid ${COLORS.info};
`;

const Select = styled.select`
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
  background: white;
  transition: border-color 0.2s ease;

  &:focus {
    outline: none;
    border-color: ${COLORS.primary};
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
`;

const RangeContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const RangeInput = styled.input`
  flex: 1;
  height: 8px;
  border-radius: 4px;
  background: #e0e0e0;
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${COLORS.primary};
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  
  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${COLORS.primary};
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
`;

const RangeValue = styled.span`
  font-weight: bold;
  font-size: 16px;
  color: ${COLORS.primary};
  min-width: 30px;
  text-align: center;
`;

const CheckboxContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 2px solid #e0e0e0;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    border-color: ${COLORS.primary};
    background: rgba(102, 126, 234, 0.05);
  }
`;

const Checkbox = styled.input`
  width: 20px;
  height: 20px;
  accent-color: ${COLORS.primary};
  cursor: pointer;
`;

const CheckboxLabel = styled.span`
  font-size: 14px;
  color: #555;
  cursor: pointer;
  flex: 1;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 24px;
`;

const Button = styled.button`
  flex: 1;
  padding: 12px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  
  ${props => props.primary ? `
    background: ${COLORS.primary};
    color: white;
    
    &:hover:not(:disabled) {
      background: #5a6fd8;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
  ` : `
    background: white;
    color: ${COLORS.primary};
    border: 2px solid ${COLORS.primary};
    
    &:hover:not(:disabled) {
      background: ${COLORS.primary};
      color: white;
    }
  `}
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
  
  &:active:not(:disabled) {
    transform: translateY(0);
  }
`;

const ErrorMessage = styled.div`
  background: #ffebee;
  color: #c62828;
  padding: 12px;
  border-radius: 8px;
  border-left: 4px solid #c62828;
  margin-top: 16px;
  font-size: 14px;
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
    onConfigChange({
      [field]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onStartGame();
  };

  return (
    <ConfigContainer>
      <ConfigTitle>
        {gameActive ? 'Current Game Settings' : 'Game Configuration'}
      </ConfigTitle>
      
      <form onSubmit={handleSubmit}>
        {/* Number of pixels */}
        <ConfigSection>
          <Label>Number of Pixels to Reveal</Label>
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
            <RangeValue>{config.k}</RangeValue>
          </RangeContainer>
        </ConfigSection>

        {/* Player role */}
        <ConfigSection>
          <Label>Your Role</Label>
          <Description>
            {getRoleDescription(config.player_role)}
          </Description>
          <Select
            value={config.player_role}
            onChange={(e) => handleChange('player_role', e.target.value)}
            disabled={gameActive}
          >
            <option value={PLAYER_ROLES.HONEST}>
              Honest Player (Help judge classify correctly)
            </option>
            <option value={PLAYER_ROLES.LIAR}>
              Liar Player (Try to mislead the judge)
            </option>
          </Select>
        </ConfigSection>

        {/* AI opponent type */}
        <ConfigSection>
          <Label>AI Opponent</Label>
          <Description>
            {getAgentDescription(config.agent_type)}
          </Description>
          <Select
            value={config.agent_type}
            onChange={(e) => handleChange('agent_type', e.target.value)}
            disabled={gameActive}
          >
            <option value={AGENT_TYPES.GREEDY}>
              Greedy AI (Fast, immediate decisions)
            </option>
            <option value={AGENT_TYPES.MCTS}>
              Strategic AI (Slower, but plans ahead)
            </option>
          </Select>
        </ConfigSection>

        {/* Precommit strategy */}
        <ConfigSection>
          <Label>Precommit Strategy</Label>
          <Description>
            When enabled, the liar agent commits to a specific wrong answer at the start, 
            potentially making it more strategic but also more predictable.
          </Description>
          <CheckboxContainer>
            <Checkbox
              type="checkbox"
              checked={config.precommit}
              onChange={(e) => handleChange('precommit', e.target.checked)}
              disabled={gameActive}
            />
            <CheckboxLabel>Enable precommit strategy</CheckboxLabel>
          </CheckboxContainer>
        </ConfigSection>

        {/* Judge model selection */}
        <ConfigSection>
          <Label>Judge Model</Label>
          <Description>
            The AI judge that will evaluate the revealed pixels and make the final classification.
          </Description>
          <Select
            value={config.judge_name}
            onChange={(e) => handleChange('judge_name', e.target.value)}
            disabled={gameActive}
          >
            <option value="28">Standard Judge (28x28 resolution)</option>
            <option value="16">Compact Judge (16x16 resolution)</option>
          </Select>
        </ConfigSection>

        {/* Pixel selection threshold */}
        <ConfigSection>
          <Label>Pixel Selection Mode</Label>
          <Description>
            Standard mode: Only allow selecting pixels with intensity > 0 (bright pixels).
            OOD mode: Allow selecting any pixel, including completely black ones (intensity = 0).
          </Description>
          <CheckboxContainer>
            <Checkbox
              type="checkbox"
              checked={config.allow_zero_pixels}
              onChange={(e) => handleChange('allow_zero_pixels', e.target.checked)}
              disabled={gameActive}
            />
            <CheckboxLabel>
              Enable OOD mode (allow selection of black pixels)
            </CheckboxLabel>
          </CheckboxContainer>
        </ConfigSection>

        {/* Action buttons */}
        <ButtonGroup>
          {gameActive ? (
            <Button
              type="button"
              onClick={onResetGame}
              disabled={loading}
            >
              New Game
            </Button>
          ) : (
            <>
              <Button
                type="submit"
                primary
                disabled={loading}
              >
                {loading ? 'Creating Game...' : 'Start Game'}
              </Button>
            </>
          )}
        </ButtonGroup>

        {/* Error display */}
        {error && (
          <ErrorMessage>
            {error}
          </ErrorMessage>
        )}
      </form>
    </ConfigContainer>
  );
};

export default GameConfig;