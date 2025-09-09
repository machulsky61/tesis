import styled from 'styled-components';
import { motion } from 'framer-motion';

// Professional card following portfolio design
export const Card = styled(motion.div)`
  background: ${props => props.theme.background.surface};
  border: 1px solid ${props => props.theme.background.border || props.theme.background.glass};
  border-radius: ${props => props.theme.radius.lg};
  box-shadow: ${props => props.theme.shadows.lg};
  padding: ${props => props.padding || '2rem'};
  transition: all 300ms ease;

  &:hover {
    transform: ${props => props.hoverable !== false ? 'translateY(-2px)' : 'none'};
    box-shadow: ${props => props.theme.shadows.xl};
    border-color: ${props => props.theme.brand.primary}20;
  }
`;

// Subtle glass card for feature highlights
export const GlassCard = styled(Card)`
  background: ${props => props.theme.background.glass};
  backdrop-filter: blur(8px);
  border: 1px solid ${props => props.theme.background.border || 'rgba(255, 255, 255, 0.1)'};
`;

// Gradient card for special emphasis
export const GradientCard = styled(Card)`
  background: ${props => props.theme.brand.gradient};
  color: ${props => props.theme.text.inverse};
  border: none;

  * {
    color: inherit;
  }
`;

// Compact card for smaller content
export const CompactCard = styled(Card)`
  padding: 1rem;
  border-radius: ${props => props.theme.radius.md};
`;

// Stats card for displaying metrics
export const StatsCard = styled(Card)`
  text-align: center;
  padding: 1rem;
  min-width: 120px;

  .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: ${props => props.theme.text.secondary};
    margin-bottom: 0.5rem;
  }

  .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: ${props => props.theme.text.primary};
    line-height: 1;
  }

  .change {
    font-size: 0.875rem;
    font-weight: 500;
    margin-top: 0.25rem;

    &.positive {
      color: ${props => props.theme.semantic.success};
    }

    &.negative {
      color: ${props => props.theme.semantic.error};
    }
  }
`;

Card.defaultProps = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.3 }
};