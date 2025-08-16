# ChisatoOPT is an experimental optimizer collection.
by win10ogod
## 🚀 Project Overview

This project is a research fork based on Eric Hartford's [GrokAdamW](https://github.com/cognitivecomputations/grokadamw), focusing on exploring innovative optimization algorithms. Building upon the original GrokAdamW, we have designed and implemented two entirely new experimental optimizers, pushing optimization theory to the frontiers of biology and mathematics.

**Note: This project is an independent research fork, unrelated to the original author.**

## 🧬 Optimizer Series

### 1. GrokAdamW (Original Implementation)
The original optimizer designed by Eric Hartford, combining Grokfast technology with AdamW to accelerate the "grokking" phenomenon in deep learning.

### 2. MathverseOpt (New Design) 🔢
**Our Original Design** - An experimental optimizer based on the Mathematical Universe Hypothesis

**Theoretical Foundation:**
- Redefines parameters as hypergraph structures H = (V, E)
- Transforms loss functions into logical inconsistency measures I(H)
- Optimizes through automated theorem proving steps
- Uses Gödel branching to handle logical incompleteness

**Core Update Equation:**
```
θ_{t+1} = θ_t - η∇I(H_θ) + Γ(P_t, G_t)
```

### 3. ViroEvoOpt (New Design) 🦠
**Our Original Design** - An optimizer based on virology, epidemiology, and evolutionary dynamics

**Biological Model Integration:**
- **SIR Epidemiological Model**: Parameters classified as Susceptible(S), Infected(I), Recovered(R)
- **Gillespie Algorithm**: Stochastic simulation of viral mutation and transmission events
- **Moran Process**: Evolutionary selection mechanism for beneficial mutations

**Viral Evolution Update:**
```
θ_I ← θ_I - η∇L(θ_I) + ξ
```
where ξ ~ Poisson(μ) · N(0,σ²) represents viral mutations


## 📊 Our Innovation Contributions

| Feature | GrokAdamW (Original) | MathverseOpt (Our Design) | ViroEvoOpt (Our Design) |
|---------|---------------------|---------------------------|-------------------------|
| **Theoretical Basis** | Grokfast + AdamW | Mathematical Universe Hypothesis | Viral Evolutionary Dynamics |
| **Parameter Representation** | Standard Tensors | Hypergraph Structure | Viral Populations |
| **Optimization Objective** | Accelerate Grokking | Logical Consistency | Evolutionary Fitness |
| **Innovation** | Slow Gradient Amplification | Theorem Proving Guidance | SIR+Gillespie+Moran |
| **Application** | Mathematical Reasoning | Logical Reasoning | Large Model Training |

## 🔬 Research Motivation and Design Philosophy

### MathverseOpt Design Motivation
Traditional optimizers treat parameters as independent numerical values, but the Mathematical Universe Hypothesis suggests that reality itself is a mathematical structure. Our design:

1. **Hypergraph Representation**: Logical relationships between parameters are more important than numerical values
2. **Consistency Optimization**: Minimize logical contradictions rather than traditional loss
3. **Proof Guidance**: Use automated theorem proving to guide parameter update directions

### ViroEvoOpt Design Motivation
Biological evolution is nature's most successful optimization process. We integrate three mature biological models:

1. **SIR Epidemiology**: Kermack-McKendrick equations (1927) describing population dynamics
2. **Gillespie Algorithm**: Exact stochastic simulation algorithm
3. **Moran Process**: Population genetics evolutionary selection model

## 🛠 Installation and Usage

### Installation
```bash
git clone https://github.com/win10ogod/ChisatoOPT
cd ChisatoOPT
pip install -e .
```

### Using Our Designed Optimizers

#### MathverseOpt Usage Example
```python
from ChisatoOPT import MathverseOpt

optimizer = MathverseOpt(
    model.parameters(),
    lr=1e-3,
    alpha=0.9,              # Hypergraph structure coefficient
    beta=0.99,              # Logical consistency coefficient  
    epsilon=1e-8,           # Numerical stability
    proof_guidance=True,    # Enable theorem proving guidance
    godel_branching=True    # Gödel branching handling
)
```

#### ViroEvoOpt Usage Example (Optimized)
```python
from ChisatoOPT import ViroEvoOpt

optimizer = ViroEvoOpt(
    model.parameters(),
    lr=1e-3,
    beta=0.1,               # Infection rate β
    gamma=0.05,             # Recovery rate γ  
    mu=1e-4,                # Mutation rate μ (based on HIV mutation rate)
    alpha=1.0,              # Loss-death rate scaling
    temperature=1.0,        # Boltzmann selection temperature
    gillespie_steps=5,      # Monte Carlo steps (optimized)
    dt=0.01                 # SIR integration time step
)

# Monitor viral evolution metrics
viral_metrics = optimizer.get_viral_metrics()
print(f"Basic reproduction number R₀: {viral_metrics['average_reproduction_number']:.3f}")
print(f"Infected population: {viral_metrics['average_infected_population']:.1f}")
print(f"Total mutations: {viral_metrics['total_viral_mutations']}")
```

## 🧮 Mathematical Foundation

### ViroEvoOpt SIR Dynamics

**SIR Differential Equation System (Kermack-McKendrick 1927):**
```
dS/dt = -β(SI/N) + μR
dI/dt = β(SI/N) - γI - δI  
dR/dt = γI - μR
```

**Basic Reproduction Number:**
```
R₀ = β / (γ + δ)
```

**Viral Mutation Generation:**
```
ξ ~ Poisson(μ) · N(0, σ²)
```

### MathverseOpt Hypergraph Theory

**Hypergraph Representation:**
```
H = (V, E) where V = parameters, E = logical relationships
```

**Consistency Optimization:**
```
I(H) = Σ inconsistency_measure(e) for e ∈ E
```

**Proof-Guided Update:**
```
θ_{t+1} = θ_t - η∇I(H) + Γ(P_t, G_t)
```

## 🎯 Usage Recommendations

### When to Use MathverseOpt
- ✅ Logical reasoning tasks
- ✅ Symbolic computation problems
- ✅ Theorem proving applications
- ✅ Cases with clear logical relationships between parameters

### When to Use ViroEvoOpt
- ✅ Large language model training
- ✅ Complex multi-modal optimization problems
- ✅ Scenarios requiring biological interpretability
- ✅ Training with limited computational resources

## 🔍 Experimental Validation

### Our Test Results
- **ViroEvoOpt**: Tested on LLaMA series models, showing significant training efficiency improvements
- **MathverseOpt**: Demonstrated unique optimization trajectories on mathematical reasoning tasks
- **Performance Comparison**: Compared with standard AdamW on multiple benchmark tests

## 📚 Citing Our Work

If you use our designed optimizers in your research, please cite:

```bibtex
@software{ChisatoOPT2025,
  title={ChisatoOPT: Novel Optimizers for Advanced AI Training},
  author={win10ogod},
  year={2024},
  url={https://github.com/win10ogod/ChisatoOPT},
  note={Fork of original GrokAdamW with MathverseOpt and ViroEvoOpt innovations}
}
```

## 🙏 Acknowledgments

- **Eric Hartford**: Creator of the original GrokAdamW project
- **Biology Research Community**: Researchers of SIR models, Gillespie algorithm, and Moran process
- **Mathematical Physics Community**: Theoretical foundation of the Mathematical Universe Hypothesis

## 📄 License

This fork project follows the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

*Exploring new frontiers in optimization algorithms: From mathematical universes to viral evolution* 🧬🔢
