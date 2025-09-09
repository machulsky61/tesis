import React from 'react';
import styled from 'styled-components';
import { motion, useScroll, useTransform } from 'framer-motion';
import Navbar from './ui/Navbar';
import { 
  ArrowDown, 
  Play, 
  BookOpen, 
  Github, 
  Mail, 
  Linkedin,
  ExternalLink,
  Download,
  Zap,
  Target,
  Users,
  Brain,
  Code,
  Cpu,
  Database,
  Globe
} from 'lucide-react';
import { Button, SecondaryButton } from './ui/Button';
import { Card, GlassCard } from './ui/Card';

const LandingContainer = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.background.primary};
  position: relative;
  overflow-x: hidden;
  padding-top: 5rem; /* Space for fixed navbar */
`;

const Section = styled(motion.section)`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  position: relative;
  
  @media (max-width: 768px) {
    padding: 1rem;
    min-height: 80vh;
  }
`;

const SectionContent = styled.div`
  max-width: 1200px;
  width: 100%;
  z-index: 2;
  position: relative;
`;

const AnimatedBackground = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  opacity: 0.03;
  z-index: 0;
  pointer-events: none;
  
  &::before {
    content: '';
    position: absolute;
    top: 20%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, ${props => props.theme.brand.primary} 0%, transparent 70%);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: 20%;
    right: 10%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, ${props => props.theme.brand.secondary} 0%, transparent 70%);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite reverse;
  }
  
  @keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
  }
`;

const HeroSection = styled(Section)`
  background: ${props => props.theme.background.primary};
  text-align: center;
  min-height: calc(100vh - 5rem); /* Account for navbar */
  padding-top: 2rem; /* Additional spacing */
  
  .hero-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2.5rem;
    max-width: 1000px;
    margin: 0 auto;
  }
  
  .hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 700;
    background: ${props => props.theme.brand.gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
  }
  
  .hero-subtitle {
    font-size: clamp(1rem, 2.5vw, 1.5rem);
    color: ${props => props.theme.text.secondary};
    font-weight: 500;
    margin: 0;
    max-width: 600px;
  }
  
  .hero-description {
    font-size: clamp(1rem, 2vw, 1.25rem);
    color: ${props => props.theme.text.muted};
    line-height: 1.6;
    max-width: 700px;
    margin: 0;
  }
  
  .hero-cta {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2rem;
    margin-top: 1rem;
  }
  
  .start-game-btn {
    font-size: 1.125rem;
    padding: 1.25rem 2.5rem;
    border-radius: 1rem;
    background: ${props => props.theme.brand.gradient};
    border: none;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 200ms ease;
    box-shadow: ${props => props.theme.shadows.lg};
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: ${props => props.theme.shadows.xl};
    }
    
    &:active {
      transform: translateY(0);
    }
  }
  
  /* Remove overlapping scroll indicator */
  .scroll-hint {
    margin-top: 1rem;
    color: ${props => props.theme.text.muted};
    font-size: 0.875rem;
    font-weight: 500;
  }
`;

const ContentSection = styled(Section)`
  min-height: auto;
  padding: 4rem 2rem;
  
  @media (max-width: 768px) {
    padding: 3rem 1rem;
  }

  .section-header {
    text-align: center;
    margin-bottom: 3rem;
    
    .section-title {
      font-size: clamp(2rem, 5vw, 3rem);
      font-weight: 700;
      color: ${props => props.theme.text.primary};
      margin: 0 0 1rem 0;
    }
    
    .section-subtitle {
      font-size: 1.125rem;
      color: ${props => props.theme.text.secondary};
      margin: 0;
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.6;
    }
  }
`;

const AboutGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-top: 3rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
`;

const FeatureCard = styled(GlassCard)`
  text-align: center;
  padding: 2rem;
  transition: all 300ms ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: ${props => props.theme.shadows.xl};
  }
  
  .feature-icon {
    width: 3rem;
    height: 3rem;
    color: ${props => props.theme.brand.primary};
    margin: 0 auto 1rem;
  }
  
  .feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: ${props => props.theme.text.primary};
    margin: 0 0 0.5rem 0;
  }
  
  .feature-description {
    color: ${props => props.theme.text.secondary};
    line-height: 1.6;
    margin: 0;
  }
`;

const ThesisShowcase = styled(Card)`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  align-items: center;
  padding: 3rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 2rem;
    padding: 2rem;
  }
  
  .thesis-content {
    .thesis-title {
      font-size: 2rem;
      font-weight: 700;
      color: ${props => props.theme.text.primary};
      margin: 0 0 1rem 0;
    }
    
    .thesis-description {
      color: ${props => props.theme.text.secondary};
      line-height: 1.6;
      margin: 0 0 2rem 0;
    }
    
    .thesis-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
      margin-bottom: 2rem;
    }
    
    .stat-item {
      text-align: center;
      padding: 1rem;
      background: ${props => props.theme.background.glass};
      border-radius: ${props => props.theme.radius.md};
      
      .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: ${props => props.theme.brand.primary};
        margin: 0;
      }
      
      .stat-label {
        font-size: 0.875rem;
        color: ${props => props.theme.text.secondary};
        margin: 0;
      }
    }
  }
  
  .thesis-visual {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    
    .thesis-preview {
      width: 200px;
      height: 280px;
      background: ${props => props.theme.background.glass};
      border-radius: ${props => props.theme.radius.lg};
      display: flex;
      align-items: center;
      justify-content: center;
      border: 2px solid ${props => props.theme.brand.primary}40;
      
      .preview-icon {
        width: 4rem;
        height: 4rem;
        color: ${props => props.theme.brand.primary};
      }
    }
  }
`;

const TechStack = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
  
  @media (max-width: 640px) {
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }
  
  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

const TechItem = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 1.5rem 1rem;
  background: ${props => props.theme.background.surface};
  border: 1px solid ${props => props.theme.background.border || props.theme.background.glass};
  border-radius: ${props => props.theme.radius.lg};
  transition: all 200ms ease;
  text-align: center;
  min-height: 120px;
  
  &:hover {
    transform: translateY(-2px);
    background: ${props => props.theme.brand.primary}08;
    border-color: ${props => props.theme.brand.primary}20;
  }
  
  .tech-icon {
    width: 2.5rem;
    height: 2.5rem;
    color: ${props => props.theme.brand.primary};
    margin: 0 auto;
  }
  
  .tech-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: ${props => props.theme.text.primary};
    margin: 0;
  }
`;

const ContactForm = styled(Card)`
  max-width: 600px;
  margin: 0 auto;
  padding: 3rem;
  
  @media (max-width: 768px) {
    padding: 2rem;
  }
  
  @media (max-width: 480px) {
    padding: 1.5rem;
  }
  
  .form-title {
    font-size: 2rem;
    font-weight: 700;
    color: ${props => props.theme.text.primary};
    margin: 0 0 2rem 0;
    text-align: center;
  }
  
  .form-group {
    margin-bottom: 1.5rem;
    
    label {
      display: block;
      font-size: 0.875rem;
      font-weight: 500;
      color: ${props => props.theme.text.primary};
      margin-bottom: 0.5rem;
    }
    
    input, textarea {
      width: 100%;
      padding: 1rem;
      border: 2px solid ${props => props.theme.background.glass};
      border-radius: ${props => props.theme.radius.md};
      background: ${props => props.theme.background.surface};
      color: ${props => props.theme.text.primary};
      font-family: inherit;
      transition: all 300ms ease;
      
      &:focus {
        outline: none;
        border-color: ${props => props.theme.brand.primary};
        box-shadow: 0 0 0 3px ${props => props.theme.brand.primary}20;
      }
      
      &::placeholder {
        color: ${props => props.theme.text.secondary};
      }
    }
    
    textarea {
      min-height: 120px;
      resize: vertical;
    }
  }
`;

const SocialLinks = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2rem;
  
  .social-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 3rem;
    height: 3rem;
    background: ${props => props.theme.background.glass};
    border-radius: ${props => props.theme.radius.full};
    color: ${props => props.theme.text.secondary};
    transition: all 300ms ease;
    text-decoration: none;
    
    &:hover {
      background: ${props => props.theme.brand.primary};
      color: white;
      transform: translateY(-3px);
    }
  }
`;

// Simplified animation variants for better performance
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: {
      duration: 0.4,
      ease: "easeOut"
    }
  }
};

const LandingPage = ({ onStartGame }) => {
  const { scrollY } = useScroll();
  // Reduce parallax effect intensity for better performance
  const backgroundY = useTransform(scrollY, [0, 2000], [0, -50]);

  return (
    <LandingContainer>
      <Navbar onStartGame={onStartGame} />
      <AnimatedBackground style={{ y: backgroundY }} />
      
      {/* Hero Section */}
      <HeroSection id="hero">
        <SectionContent>
          <motion.div
            className="hero-content"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <motion.h1 
              className="hero-title"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              AI Safety via Debate
            </motion.h1>
            
            <motion.h2 
              className="hero-subtitle"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              Interactive Research Demonstration on MNIST Classification
            </motion.h2>
            
            <motion.p 
              className="hero-description"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              Exploring AI alignment through strategic adversarial interactions. A Data Science thesis project implementing debate methodology for neural network interpretability and safety research.
            </motion.p>
            
            <motion.div 
              className="hero-cta"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <motion.button
                className="start-game-btn"
                onClick={onStartGame}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Play size={20} />
                Try Interactive Demo
              </motion.button>
              
              <div className="scroll-hint">
                Scroll down to explore the research
              </div>
            </motion.div>
          </motion.div>
        </SectionContent>
      </HeroSection>

      {/* Research Section */}
      <ContentSection id="research">
        <SectionContent>
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">Research Methodology</h2>
            <p className="section-subtitle">
              Implementing "AI Safety via Debate" for neural network interpretability and alignment research
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <AboutGrid>
              <motion.div variants={itemVariants}>
                <FeatureCard>
                  <Brain className="feature-icon" />
                  <h3 className="feature-title">Adversarial Debate Framework</h3>
                  <p className="feature-description">
                    Two AI agents with opposing objectives compete to convince a neural judge through strategic pixel revelation on MNIST digits
                  </p>
                </FeatureCard>
              </motion.div>

              <motion.div variants={itemVariants}>
                <FeatureCard>
                  <Target className="feature-icon" />
                  <h3 className="feature-title">Neural Network Interpretability</h3>
                  <p className="feature-description">
                    SparseCNN judge trained on masked images to study decision-making processes and model behavior under incomplete information
                  </p>
                </FeatureCard>
              </motion.div>

              <motion.div variants={itemVariants}>
                <FeatureCard>
                  <Zap className="feature-icon" />
                  <h3 className="feature-title">AI Alignment Research</h3>
                  <p className="feature-description">
                    Exploring truthfulness, deception, and strategic behavior in AI systems through controlled adversarial interactions
                  </p>
                </FeatureCard>
              </motion.div>
            </AboutGrid>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* Thesis Section */}
      <ContentSection id="thesis">
        <SectionContent>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <ThesisShowcase>
              <div className="thesis-content">
                <h2 className="thesis-title">Master's Thesis</h2>
                <p className="thesis-description">
                  This interactive game is part of my Data Science Master's thesis, implementing and analyzing "AI Safety via Debate" methodology on the MNIST dataset to study argumentative strategies and judge bias effects.
                </p>
                
                <div className="thesis-stats">
                  <div className="stat-item">
                    <div className="stat-value">10K+</div>
                    <div className="stat-label">Experiments</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-value">95%</div>
                    <div className="stat-label">Accuracy</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-value">4</div>
                    <div className="stat-label">AI Agents</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-value">28x28</div>
                    <div className="stat-label">Pixel Grid</div>
                  </div>
                </div>
                
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <Button>
                    <Download style={{ width: '1rem', height: '1rem' }} />
                    Download Thesis
                  </Button>
                  <SecondaryButton>
                    <ExternalLink style={{ width: '1rem', height: '1rem' }} />
                    View Results
                  </SecondaryButton>
                </div>
              </div>
              
              <div className="thesis-visual">
                <div className="thesis-preview">
                  <BookOpen className="preview-icon" />
                </div>
                <p style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  Full thesis document with methodology, results, and analysis
                </p>
              </div>
            </ThesisShowcase>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* How It Works Section */}
      <ContentSection id="methodology" style={{ background: `${props => props.theme.background.muted}` }}>
        <SectionContent>
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">How It Works</h2>
            <p className="section-subtitle">
              Step-by-step breakdown of the AI safety debate methodology
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginTop: '3rem' }}
          >
            <motion.div variants={itemVariants}>
              <FeatureCard>
                <div className="feature-icon" style={{ background: 'oklch(0.6 0.2 240 / 0.1)', color: 'oklch(0.6 0.2 240)' }}>1</div>
                <h3 className="feature-title">Image Presentation</h3>
                <p className="feature-description">
                  A 28x28 MNIST digit is shown to two competing AI agents and a neural judge, with all pixels initially hidden behind masks.
                </p>
              </FeatureCard>
            </motion.div>

            <motion.div variants={itemVariants}>
              <FeatureCard>
                <div className="feature-icon" style={{ background: 'oklch(0.6 0.2 120 / 0.1)', color: 'oklch(0.6 0.2 120)' }}>2</div>
                <h3 className="feature-title">Strategic Revelation</h3>
                <p className="feature-description">
                  Agents take turns revealing pixels strategically - the honest agent tries to help classification, while the liar tries to mislead.
                </p>
              </FeatureCard>
            </motion.div>

            <motion.div variants={itemVariants}>
              <FeatureCard>
                <div className="feature-icon" style={{ background: 'oklch(0.577 0.245 27.325 / 0.1)', color: 'oklch(0.577 0.245 27.325)' }}>3</div>
                <h3 className="feature-title">Judge Decision</h3>
                <p className="feature-description">
                  After k pixel revelations, a SparseCNN judge trained on masked images makes the final classification decision based on revealed evidence.
                </p>
              </FeatureCard>
            </motion.div>

            <motion.div variants={itemVariants}>
              <FeatureCard>
                <div className="feature-icon" style={{ background: 'oklch(0.556 0.15 280 / 0.1)', color: 'oklch(0.556 0.15 280)' }}>4</div>
                <h3 className="feature-title">Analysis</h3>
                <p className="feature-description">
                  Results are analyzed to study truthfulness, deception strategies, judge bias, and the effectiveness of debate for AI alignment.
                </p>
              </FeatureCard>
            </motion.div>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* Research Impact Section */}
      <ContentSection id="impact">
        <SectionContent>
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">Research Impact</h2>
            <p className="section-subtitle">
              Key findings and contributions to AI safety research
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
            style={{ margin: '3rem 0' }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
              <Card style={{ textAlign: 'center', padding: '2rem' }}>
                <div style={{ fontSize: '3rem', fontWeight: '700', color: 'oklch(0.6 0.2 240)', marginBottom: '0.5rem' }}>85%</div>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Judge Accuracy</h4>
                <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                  SparseCNN achieves 85% accuracy on masked MNIST images, demonstrating effective learning from partial information
                </p>
              </Card>
              
              <Card style={{ textAlign: 'center', padding: '2rem' }}>
                <div style={{ fontSize: '3rem', fontWeight: '700', color: 'oklch(0.6 0.2 120)', marginBottom: '0.5rem' }}>73%</div>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Honest Win Rate</h4>
                <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                  Honest agents successfully convince judges more often, supporting truth-conducive properties of debate
                </p>
              </Card>
              
              <Card style={{ textAlign: 'center', padding: '2rem' }}>
                <div style={{ fontSize: '3rem', fontWeight: '700', color: 'oklch(0.577 0.245 27.325)', marginBottom: '0.5rem' }}>4</div>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Agent Types</h4>
                <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                  Greedy, MCTS, and adversarial agents tested to understand strategic behavior in debate scenarios
                </p>
              </Card>
            </div>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* Demo Section */}
      <ContentSection id="demo" style={{ background: `${props => props.theme.brand.secondary}05` }}>
        <SectionContent>
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">Interactive Demonstration</h2>
            <p className="section-subtitle">
              Experience the debate methodology firsthand through our interactive web interface
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
            style={{ textAlign: 'center', margin: '3rem 0' }}
          >
            <Card style={{ maxWidth: '700px', margin: '0 auto', textAlign: 'center', padding: '3rem 2rem' }}>
              <h3 style={{ marginBottom: '1.5rem', color: 'var(--text-primary)', fontSize: '1.5rem' }}>
                Try the AI Safety Debate Game
              </h3>
              <p style={{ marginBottom: '2rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                Play as the honest or deceptive agent, strategically revealing pixels to influence 
                the neural judge's decision on MNIST digit classification. Experience firsthand how 
                debate mechanisms can enhance AI interpretability and safety.
              </p>
              <Button onClick={onStartGame} style={{ fontSize: '1.125rem', padding: '1.25rem 2.5rem', marginBottom: '1.5rem' }}>
                <Play size={20} />
                Launch Interactive Demo
              </Button>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap', marginTop: '1.5rem' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600', color: 'oklch(0.556 0.15 280)' }}>28×28</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Pixel Grid</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600', color: 'oklch(0.556 0.15 280)' }}>10K+</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Test Images</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600', color: 'oklch(0.556 0.15 280)' }}>4</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>AI Agents</div>
                </div>
              </div>
            </Card>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* Tech Section */}
      <ContentSection id="tech">
        <SectionContent>
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">Technical Implementation</h2>
            <p className="section-subtitle">
              Modern full-stack architecture designed for AI research and interactive demonstrations
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <TechStack>
              <TechItem variants={itemVariants}>
                <Brain className="tech-icon" />
                <div className="tech-name">PyTorch</div>
              </TechItem>
              <TechItem variants={itemVariants}>
                <Cpu className="tech-icon" />
                <div className="tech-name">NumPy</div>
              </TechItem>
              <TechItem variants={itemVariants}>
                <Code className="tech-icon" />
                <div className="tech-name">React</div>
              </TechItem>
              <TechItem variants={itemVariants}>
                <Database className="tech-icon" />
                <div className="tech-name">FastAPI</div>
              </TechItem>
              <TechItem variants={itemVariants}>
                <Globe className="tech-icon" />
                <div className="tech-name">Framer Motion</div>
              </TechItem>
              <TechItem variants={itemVariants}>
                <Zap className="tech-icon" />
                <div className="tech-name">Styled Components</div>
              </TechItem>
            </TechStack>
            
            <motion.div 
              variants={itemVariants} 
              style={{ textAlign: 'center', marginTop: '3rem' }}
            >
              <Button onClick={() => window.open('https://github.com/machulsky61/tesis', '_blank')}>
                <Github style={{ width: '1rem', height: '1rem' }} />
                View Source Code
              </Button>
            </motion.div>
          </motion.div>
        </SectionContent>
      </ContentSection>

      {/* Simplified Contact Section */}
      <ContentSection id="contact" style={{ background: `${props => props.theme.background.muted}` }}>
        <SectionContent>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            style={{ textAlign: 'center' }}
          >
            <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
              Connect & Collaborate
            </h2>
            <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: '3rem', maxWidth: '600px', margin: '0 auto 3rem' }}>
              Interested in AI safety research, neural network interpretability, or collaboration opportunities?
            </p>
            
            <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
              <Button onClick={() => window.open('https://github.com/machulsky61', '_blank')} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Github size={18} />
                GitHub
              </Button>
              <Button onClick={() => window.open('https://linkedin.com/in/joaquin-machulsky', '_blank')} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Linkedin size={18} />
                LinkedIn  
              </Button>
              <Button onClick={() => window.location.href = 'mailto:joaquinmachulsky@gmail.com'} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Mail size={18} />
                Email
              </Button>
            </div>
            
            <p style={{ marginTop: '3rem', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
              © 2025 Joaquin Salvador Machulsky. All rights reserved.
            </p>
          </motion.div>
        </SectionContent>
      </ContentSection>
    </LandingContainer>
  );
};

export default LandingPage;