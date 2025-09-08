import styled from 'styled-components';
import { motion } from 'framer-motion';

// Base card with glassmorphism effect
export const Card = styled(motion.div)`
  background: ${props => props.theme.background.surface};
  backdrop-filter: ${props => props.theme.blur.md};
  border: 1px solid ${props => props.theme.background.glass};
  border-radius: ${props => props.theme.radius.lg};
  box-shadow: ${props => props.theme.shadows.glass};
  padding: ${props => props.padding || '1.5rem'};
  transition: all 300ms cubic-bezier(0.175, 0.885, 0.32, 1.275);

  &:hover {
    transform: ${props => props.hoverable !== false ? 'translateY(-2px)' : 'none'};
    box-shadow: ${props => props.theme.shadows.xl};
    border-color: ${props => props.theme.brand.primary}40;
  }
`;

// Glass card with more transparency
export const GlassCard = styled(Card)`
  background: ${props => props.theme.background.glass};
  backdrop-filter: ${props => props.theme.blur.lg};
  border: 1px solid ${props => props.theme.background.glass};
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