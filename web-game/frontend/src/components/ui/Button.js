import styled from 'styled-components';
import { motion } from 'framer-motion';

// Professional button following portfolio design
export const Button = styled(motion.button)`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: ${props => props.theme.radius.md};
  font-size: 0.875rem;
  font-weight: 600;
  font-family: inherit;
  text-decoration: none;
  cursor: pointer;
  transition: all 200ms ease;
  position: relative;
  
  /* Base styles - Clean minimal approach */
  background: ${props => props.theme.brand.primary};
  color: ${props => props.theme.text.inverse};
  box-shadow: ${props => props.theme.shadows.md};

  /* Subtle hover effects */
  &:hover:not(:disabled) {
    background: ${props => props.theme.brand.secondary};
    transform: translateY(-1px);
    box-shadow: ${props => props.theme.shadows.lg};
  }

  /* Active state */
  &:active:not(:disabled) {
    transform: translateY(0);
  }

  /* Focus state */
  &:focus-visible {
    outline: 2px solid ${props => props.theme.brand.primary};
    outline-offset: 2px;
  }

  /* Disabled state */
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

// Variant: Secondary button - Outline style
export const SecondaryButton = styled(Button)`
  background: transparent;
  color: ${props => props.theme.brand.primary};
  border: 1px solid ${props => props.theme.background.border || props.theme.brand.primary};
  box-shadow: none;

  &:hover:not(:disabled) {
    background: ${props => props.theme.brand.primary}10;
    color: ${props => props.theme.brand.primary};
    border-color: ${props => props.theme.brand.primary};
    box-shadow: ${props => props.theme.shadows.sm};
  }
`;

// Variant: Ghost button
export const GhostButton = styled(Button)`
  background: transparent;
  color: ${props => props.theme.text.primary};
  box-shadow: none;

  &:hover:not(:disabled) {
    background: ${props => props.theme.interactive.hover};
    color: ${props => props.theme.brand.primary};
    transform: none;
    box-shadow: none;
  }
`;

// Variant: Success button
export const SuccessButton = styled(Button)`
  background: ${props => props.theme.semantic.success};

  &:hover:not(:disabled) {
    background: ${props => props.theme.semantic.success}dd;
  }
`;

// Variant: Error button
export const ErrorButton = styled(Button)`
  background: ${props => props.theme.semantic.error};

  &:hover:not(:disabled) {
    background: ${props => props.theme.semantic.error}dd;
  }
`;

// Variant: Icon button
export const IconButton = styled(Button)`
  padding: 0.75rem;
  min-width: 0;
  width: 2.75rem;
  height: 2.75rem;
  border-radius: ${props => props.theme.radius.full};

  svg {
    width: 1.25rem;
    height: 1.25rem;
  }
`;

// Variant: Large button
export const LargeButton = styled(Button)`
  padding: 1rem 2rem;
  font-size: 1rem;
  border-radius: ${props => props.theme.radius.lg};
`;

// Variant: Small button
export const SmallButton = styled(Button)`
  padding: 0.5rem 1rem;
  font-size: 0.75rem;
  border-radius: ${props => props.theme.radius.sm};
`;

// Button group container
export const ButtonGroup = styled.div`
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  align-items: center;
  justify-content: ${props => props.justify || 'flex-start'};

  @media (max-width: 640px) {
    flex-direction: column;
    width: 100%;

    button {
      width: 100%;
    }
  }
`;

// Loading button variant
export const LoadingButton = styled(Button)`
  &:disabled {
    opacity: 1;
    cursor: wait;
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

// Floating Action Button
export const FAB = styled(IconButton)`
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  width: 3.5rem;
  height: 3.5rem;
  z-index: 1000;
  box-shadow: ${props => props.theme.shadows.xl};

  &:hover:not(:disabled) {
    transform: translateY(-3px) scale(1.05);
    box-shadow: ${props => props.theme.shadows.xl};
  }

  @media (max-width: 640px) {
    bottom: 1rem;
    right: 1rem;
    width: 3rem;
    height: 3rem;
  }
`;

// Default animation props
Button.defaultProps = {
  whileTap: { scale: 0.98 },
  transition: { type: 'spring', stiffness: 300, damping: 20 }
};