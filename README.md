# Evaluating Argumentative Strategies in AI Debates

**Data Science Master's Thesis (UBA)** investigating argumentative strategies in debates between AI systems with asymmetric capabilities and cognitively limited judges.

## 🎯 Research Focus

This thesis explores how unrestricted AI agents exploit deceptive tactics to win debates against honest agents, and analyzes the impact of lower-capacity judges evaluating arguments from more advanced agents.

### Key Research Questions

- How do asymmetric agent capabilities affect debate outcomes?
- What argumentative strategies emerge under different conditions?
- How does judge bias impact convergence towards truth?
- Can debate serve as an effective AI alignment technique?

## 🧪 Implementation: AI Safety via Debate on MNIST

The [`debate_mnist/`](debate_mnist/) directory contains a complete implementation replicating Irving et al. (2018) with additional extensions:

- **Asymmetric Debates**: MCTS vs Greedy agent competitions
- **Judge Evaluation**: 8 different pixel selection strategies
- **Bias Analysis**: Precommit strategies and adversarial evaluation
- **Logits Tracking**: Progressive judge decision-making analysis

### Quick Start

```bash
cd debate_mnist
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_experiments.py  # Complete automation
```

## 📊 Key Contributions

1. **Asymmetric Agent Analysis**: First systematic study of mixed-capability debates
2. **Judge Bias Quantification**: Novel metrics for evaluating judge robustness
3. **Strategic Behavior Emergence**: Documentation of deceptive patterns in pixel selection
4. **Scalable Evaluation Framework**: Automated experimentation with comprehensive logging

## 📚 Theoretical Foundation

Based on recent advances in AI safety and alignment:

- **AI Safety via Debate** (Irving et al., 2018) - Core methodology
- **Scalable AI Safety via Doubly-Efficient Debate** (Khan et al., 2023)
- **Measuring Progress on Scalable Oversight** (Bowman et al., 2022)
- **AI Control: Improving Safety Despite Intentional Subversion** (Greenblatt et al., 2023)

## 📈 Expected Outcomes

- Quantitative analysis of debate dynamics under capability asymmetry
- Taxonomy of emergent argumentative strategies
- Framework for evaluating judge bias in AI systems
- Recommendations for debate-based alignment techniques

## 📬 Contact

**Joaquín Salvador Machulsky**
Email: [jmachulsky@dc.uba.ar](mailto:jmachulsky@dc.uba.ar)
