import React from 'react';
import styled from 'styled-components';
import { COLORS } from '../utils/constants';
import { formatDuration, formatPercentage, getStrategyDescription } from '../utils/helpers';

const ResultContainer = styled.div`
  background: ${COLORS.cardBackground};
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  max-width: 500px;
  width: 100%;
  text-align: center;
`;

const ResultTitle = styled.h2`
  margin: 0 0 24px 0;
  font-size: 24px;
  font-weight: 700;
  color: ${props => props.won ? COLORS.success : COLORS.error};
`;

const ResultIcon = styled.div`
  font-size: 64px;
  margin-bottom: 16px;
  color: ${props => props.won ? COLORS.success : COLORS.error};
`;

const MainStats = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 24px 0;
`;

const StatCard = styled.div`
  background: #f8f9fa;
  padding: 16px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.color || COLORS.primary};
  
  .label {
    font-size: 12px;
    color: #666;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 8px;
  }
  
  .value {
    font-size: 24px;
    font-weight: bold;
    color: #333;
  }
  
  .subtext {
    font-size: 12px;
    color: #777;
    margin-top: 4px;
  }
`;

const DetailedStats = styled.div`
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid #e0e0e0;
`;

const StatsSection = styled.div`
  margin-bottom: 20px;
  
  .section-title {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-bottom: 12px;
    text-align: left;
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  font-size: 14px;
`;

const StatRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: white;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  
  .stat-label {
    color: #666;
    font-weight: 500;
  }
  
  .stat-value {
    color: #333;
    font-weight: 600;
  }
`;

const ActionButtons = styled.div`
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
    
    &:hover {
      background: #5a6fd8;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
  ` : `
    background: white;
    color: ${COLORS.primary};
    border: 2px solid ${COLORS.primary};
    
    &:hover {
      background: ${COLORS.primary};
      color: white;
    }
  `}
  
  &:active {
    transform: translateY(0);
  }
`;

const GameResult = ({
  gameResult,
  gameState,
  onNewGame,
  onViewMoves
}) => {
  if (!gameResult || !gameState) {
    return null;
  }

  const {
    human_won,
    final_accuracy,
    total_turns,
    game_duration,
    human_strategy_summary,
    ai_strategy_summary
  } = gameResult;

  const { config, judge_prediction, true_label } = gameState;

  // Calculate additional stats
  const humanMovesCount = human_strategy_summary.total_moves;
  const aiMovesCount = ai_strategy_summary.total_moves;
  const judgeWasCorrect = final_accuracy === 1.0;
  
  // Strategy descriptions
  const humanStrategy = getStrategyDescription(
    { x: human_strategy_summary.avg_x_position, y: human_strategy_summary.avg_y_position }
  );
  const aiStrategy = getStrategyDescription(
    { x: ai_strategy_summary.avg_x_position, y: ai_strategy_summary.avg_y_position }
  );

  return (
    <ResultContainer>
      <ResultIcon won={human_won}>
        {human_won ? '🎉' : '🤖'}
      </ResultIcon>
      
      <ResultTitle won={human_won}>
        {human_won ? 'You Won!' : 'AI Won!'}
      </ResultTitle>
      
      <p style={{ color: '#666', marginBottom: '24px' }}>
        {human_won 
          ? `Congratulations! You successfully ${config.player_role === 'honest' ? 'helped the judge classify correctly' : 'misled the judge'}.`
          : `The AI ${config.player_role === 'honest' ? 'prevented correct classification' : 'helped the judge classify correctly'} this time.`
        }
      </p>

      <MainStats>
        <StatCard color={judgeWasCorrect ? COLORS.success : COLORS.error}>
          <div className="label">Judge Prediction</div>
          <div className="value">{judge_prediction?.predicted_class || 'N/A'}</div>
          <div className="subtext">
            True label: {true_label} {judgeWasCorrect ? '✓' : '✗'}
          </div>
        </StatCard>
        
        <StatCard color={COLORS.info}>
          <div className="label">Final Confidence</div>
          <div className="value">
            {judge_prediction ? formatPercentage(judge_prediction.confidence) : 'N/A'}
          </div>
          <div className="subtext">
            Judge certainty
          </div>
        </StatCard>
      </MainStats>

      <DetailedStats>
        <StatsSection>
          <div className="section-title">Game Summary</div>
          <StatsGrid>
            <StatRow>
              <span className="stat-label">Duration</span>
              <span className="stat-value">{formatDuration(game_duration)}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">Total Moves</span>
              <span className="stat-value">{total_turns}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">Your Role</span>
              <span className="stat-value">{config.player_role}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">AI Type</span>
              <span className="stat-value">{config.agent_type}</span>
            </StatRow>
          </StatsGrid>
        </StatsSection>

        <StatsSection>
          <div className="section-title">Strategy Analysis</div>
          <StatsGrid>
            <StatRow>
              <span className="stat-label">Your Strategy</span>
              <span className="stat-value">{humanStrategy}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">AI Strategy</span>
              <span className="stat-value">{aiStrategy}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">Your Moves</span>
              <span className="stat-value">{humanMovesCount}</span>
            </StatRow>
            <StatRow>
              <span className="stat-label">AI Moves</span>
              <span className="stat-value">{aiMovesCount}</span>
            </StatRow>
          </StatsGrid>
        </StatsSection>

        {config.precommit && (
          <StatsSection>
            <div className="section-title">Special Rules</div>
            <StatRow>
              <span className="stat-label">Precommit Strategy</span>
              <span className="stat-value">Enabled</span>
            </StatRow>
          </StatsSection>
        )}
      </DetailedStats>

      <ActionButtons>
        <Button onClick={onNewGame} primary>
          Play Again
        </Button>
        {onViewMoves && (
          <Button onClick={onViewMoves}>
            View Moves
          </Button>
        )}
      </ActionButtons>
    </ResultContainer>
  );
};

export default GameResult;