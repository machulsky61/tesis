import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Moon, Sun, Menu, X } from 'lucide-react';
import { useTheme } from '../../theme/ThemeProvider';

const NavContainer = styled(motion.nav)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: ${props => props.theme.background.surface}95;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid ${props => props.theme.background.border || 'rgba(255, 255, 255, 0.1)'};
  transition: all 300ms ease;
`;

const NavContent = styled.div`
  max-width: 6rem;
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  
  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const Logo = styled.div`
  font-size: 1.25rem;
  font-weight: 700;
  color: ${props => props.theme.text.primary};
  text-decoration: none;
  cursor: pointer;
  
  .highlight {
    color: ${props => props.theme.brand.primary};
  }
`;

const NavLinks = styled.div`
  display: flex;
  align-items: center;
  gap: 2rem;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const NavLink = styled.a`
  color: ${props => props.theme.text.secondary};
  text-decoration: none;
  font-weight: 500;
  font-size: 0.875rem;
  transition: color 200ms ease;
  cursor: pointer;
  position: relative;
  
  &:hover {
    color: ${props => props.theme.brand.primary};
  }
  
  &.active {
    color: ${props => props.theme.brand.primary};
    
    &::after {
      content: '';
      position: absolute;
      bottom: -0.5rem;
      left: 0;
      right: 0;
      height: 2px;
      background: ${props => props.theme.brand.primary};
    }
  }
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const ThemeToggle = styled(motion.button)`
  width: 2.5rem;
  height: 2.5rem;
  border-radius: ${props => props.theme.radius.md};
  border: 1px solid ${props => props.theme.background.border || 'transparent'};
  background: ${props => props.theme.background.surface};
  color: ${props => props.theme.text.primary};
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 200ms ease;
  
  &:hover {
    background: ${props => props.theme.brand.primary}10;
    border-color: ${props => props.theme.brand.primary}20;
  }
`;

const MobileMenuButton = styled(motion.button)`
  display: none;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: ${props => props.theme.radius.md};
  border: 1px solid ${props => props.theme.background.border || 'transparent'};
  background: ${props => props.theme.background.surface};
  color: ${props => props.theme.text.primary};
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 200ms ease;
  
  @media (max-width: 768px) {
    display: flex;
  }
  
  &:hover {
    background: ${props => props.theme.brand.primary}10;
  }
`;

const MobileMenu = styled(motion.div)`
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: ${props => props.theme.background.surface}95;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid ${props => props.theme.background.border || 'rgba(255, 255, 255, 0.1)'};
  padding: 1rem 2rem 2rem;
  
  @media (min-width: 769px) {
    display: none !important;
  }
`;

const MobileNavLink = styled(NavLink)`
  display: block;
  padding: 0.75rem 0;
  font-size: 1rem;
  border-bottom: 1px solid ${props => props.theme.background.border || 'rgba(255, 255, 255, 0.05)'};
  
  &:last-child {
    border-bottom: none;
  }
`;

const sections = [
  { id: 'hero', label: 'Home' },
  { id: 'demo', label: 'Demo' },
  { id: 'research', label: 'Research' },
  { id: 'thesis', label: 'Thesis' },
  { id: 'methodology', label: 'Method' },
  { id: 'impact', label: 'Impact' },
  { id: 'tech', label: 'Tech' }
];

const Navbar = ({ onStartGame }) => {
  const { theme, toggleTheme } = useTheme();
  const [activeSection, setActiveSection] = useState('hero');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll to update active section
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;
      setScrolled(window.scrollY > 50);
      
      // Find active section based on scroll position
      const sectionElements = sections.map(section => 
        document.getElementById(section.id)
      ).filter(Boolean);
      
      let currentSection = 'hero';
      for (const element of sectionElements) {
        if (element.offsetTop <= scrollPosition) {
          currentSection = element.id;
        }
      }
      
      setActiveSection(currentSection);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId) => {
    if (sectionId === 'demo') {
      onStartGame?.();
      return;
    }
    
    const element = document.getElementById(sectionId);
    if (element) {
      const offset = 80; // Account for fixed navbar height
      const elementPosition = element.offsetTop - offset;
      
      window.scrollTo({
        top: elementPosition,
        behavior: 'smooth'
      });
    }
    
    setMobileMenuOpen(false);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setActiveSection('hero');
  };

  return (
    <NavContainer
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <NavContent>
        <Logo onClick={scrollToTop}>
          AI <span className="highlight">Safety</span> Research
        </Logo>
        
        <NavLinks>
          {sections.map((section) => (
            <NavLink
              key={section.id}
              className={activeSection === section.id ? 'active' : ''}
              onClick={() => scrollToSection(section.id)}
            >
              {section.label}
            </NavLink>
          ))}
        </NavLinks>
        
        <NavActions>
          <ThemeToggle
            onClick={toggleTheme}
            whileTap={{ scale: 0.95 }}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
          </ThemeToggle>
          
          <MobileMenuButton
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            whileTap={{ scale: 0.95 }}
          >
            {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </MobileMenuButton>
        </NavActions>
      </NavContent>
      
      {mobileMenuOpen && (
        <MobileMenu
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
        >
          {sections.map((section) => (
            <MobileNavLink
              key={section.id}
              className={activeSection === section.id ? 'active' : ''}
              onClick={() => scrollToSection(section.id)}
            >
              {section.label}
            </MobileNavLink>
          ))}
        </MobileMenu>
      )}
    </NavContainer>
  );
};

export default Navbar;