# Ms. Pac-Man Reinforcement Learning Project

This project implements three state-of-the-art reinforcement learning algorithms to train AI agents to play Ms. Pac-Man: **DQN (Deep Q-Network)**, **PPO (Proximal Policy Optimization)**, and **A2C (Advantage Actor-Critic)**.

## Project Overview

**Primary Goals:**
- Compare the performance of different RL algorithms on a classic Atari game
- Demonstrate fundamental and advanced RL concepts
- Provide an educational resource for learning deep reinforcement learning

**Learning Outcomes:**
- Deep reinforcement learning implementation
- Policy gradient methods (PPO, A2C)
- Value-based methods (DQN)
- Atari environment preprocessing
- Neural network design for RL
- Training visualization and debugging

## Problem Statement

### The Ms. Pac-Man Challenge

Ms. Pac-Man presents a challenging reinforcement learning problem:

**Key Challenges:**
- **High-dimensional visual input**: 210x160 RGB images (62,400 pixels)
- **Sparse rewards**: Points only awarded when eating pellets or ghosts
- **Long-horizon planning**: Episodes can last hundreds of steps
- **Partial observability**: Cannot see entire maze at once
- **Multiple objectives**: Eat pellets, avoid ghosts, eat power pellets strategically

### Why It's Difficult

1. **Credit Assignment**: When should the agent receive credit for actions taken much earlier?
2. **Exploration vs Exploitation**: Balance exploring new strategies vs exploiting known good actions
3. **Temporal Correlations**: Consecutive frames are highly similar, making learning harder
4. **Non-stationarity**: Ghost behavior changes based on game state

### Success Criteria

| Level | Average Score | Description |
|-------|--------------|-------------|
| Baseline | ~200 | Random agent |
| Target | 2000+ | Good performance |
| Stretch | 5000+ | Human-level performance |

## Algorithm Deep-Dives

### DQN (Deep Q-Network)

**Core Concept:** Approximate the action-value function Q(s,a) using a deep neural network.

**Key Innovations:**
- **Experience Replay**: Store transitions in a buffer and sample randomly to break temporal correlations
- **Target Network**: Use a separate, slowly-updated network for computing target Q-values
- **Frame Stacking**: Stack 4 consecutive frames to provide temporal information

**Loss Function:**
```
L = MSE(Q(s,a), r + γ * max_a' Q_target(s',a'))
```

**Exploration Strategy:**
- Epsilon-greedy with linear decay
- Start: ε = 1.0 (fully random)
- End: ε = 0.01 (mostly greedy)

**Network Architecture:**
```
Input (4, 84, 84) → Conv2D(32, 8x8, stride 4) → ReLU
                  → Conv2D(64, 4x4, stride 2) → ReLU
                  → Conv2D(64, 3x3, stride 1) → ReLU
                  → Flatten → FC(512) → ReLU
                  → FC(n_actions)
```

**Hyperparameters:**
- Learning rate: 1e-4
- Discount factor γ: 0.99
- Replay buffer size: 100,000
- Batch size: 32
- Target update frequency: 1,000 steps
- Training starts: 50,000 steps

### PPO (Proximal Policy Optimization)

**Core Concept:** Constrained policy gradient that prevents destructively large policy updates.

**Key Innovations:**
- **Clipped Surrogate Objective**: Limit policy updates to a trust region
- **Multiple Epochs**: Reuse collected data for multiple gradient steps
- **Generalized Advantage Estimation (GAE)**: Better advantage estimates

**Loss Function:**
```
L = L_policy + c1 * L_value + c2 * H(π)

L_policy = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
where r_t = π_new(a|s) / π_old(a|s)
```

**Network Architecture:**
```
Shared CNN Backbone:
Input (4, 84, 84) → Conv2D(32, 8x8, stride 4) → ReLU
                  → Conv2D(64, 4x4, stride 2) → ReLU
                  → Conv2D(64, 3x3, stride 1) → ReLU
                  → Flatten → FC(512) → ReLU

Actor Head:  FC(512) → FC(n_actions) → Softmax
Critic Head: FC(512) → FC(1)
```

**Hyperparameters:**
- Learning rate: 2.5e-4
- Discount factor γ: 0.99
- GAE lambda λ: 0.95
- Clip range ε: 0.2
- Epochs per update: 4
- Steps per update: 128
- Parallel environments: 8
- Entropy coefficient: 0.01
- Value function coefficient: 0.5

### A2C (Advantage Actor-Critic)

**Core Concept:** Synchronous actor-critic with advantage function to reduce variance.

**Key Components:**
- **Actor (Policy)**: π(a|s) - outputs action probabilities
- **Critic (Value)**: V(s) - outputs state value estimate
- **Advantage**: A(s,a) = Q(s,a) - V(s) ≈ R - V(s)

**Loss Function:**
```
L = L_policy + c1 * L_value + c2 * H(π)

L_policy = -log π(a|s) * A(s,a)
L_value = (V(s) - R)²
```

**Architecture:** Same as PPO (shared CNN + actor/critic heads)

**Hyperparameters:**
- Learning rate: 7e-4
- Discount factor γ: 0.99
- GAE lambda λ: 1.0 (n-step returns, no GAE)
- N-steps: 5
- Parallel environments: 8
- Entropy coefficient: 0.01
- Value function coefficient: 0.5
- Optimizer: RMSprop

## Technical Implementation

### Environment Preprocessing Pipeline

Standard Atari preprocessing for all algorithms:

1. **Frame Skip (4 frames)**: Take action for 4 consecutive frames, return max
2. **Grayscale Conversion**: RGB → Grayscale (reduces input from 3 to 1 channel)
3. **Resize**: 210x160 → 84x84 (reduces computation)
4. **Frame Stacking**: Stack 4 frames → 84x84x4 input (provides motion information)
5. **Reward Clipping**: Clip rewards to [-1, 1] for stability
6. **Episodic Life**: Treat life loss as episode end for faster learning
7. **NoOp Reset**: Random initial no-ops for state diversity

### Training Infrastructure

**Hardware Requirements:**
- GPU: Recommended (CUDA-capable with 8GB+ VRAM)
- CPU: Fallback supported (slower, ~10x)
- RAM: 16GB+ recommended
- Storage: 5GB+ for models and logs

**Training Time Estimates** (1M steps):
- GPU (RTX 3080): 2-4 hours
- CPU (8 cores): 20-40 hours

**Software Stack:**
- Python 3.8+
- PyTorch 2.0+ (deep learning framework)
- Gymnasium (OpenAI Gym successor)
- ALE-py (Arcade Learning Environment)
- OpenCV (image preprocessing)
- TensorBoard (training visualization)

## Evaluation Metrics

**Primary Metrics:**
- **Average Episode Reward**: Mean reward over last 100 episodes
- **Max Episode Reward**: Best episode reward achieved
- **Episode Length**: Number of steps per episode

**Secondary Metrics:**
- **Training Stability**: Reward variance and standard deviation
- **Sample Efficiency**: Reward per environment step
- **Convergence Speed**: Steps to reach performance threshold

## Project Structure

```
pacman/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── configs/                     # Algorithm configurations
│   ├── dqn_config.yaml         # DQN hyperparameters
│   ├── ppo_config.yaml         # PPO hyperparameters
│   └── a2c_config.yaml         # A2C hyperparameters
├── src/                         # Source code
│   ├── agents/                 # RL agent implementations
│   │   ├── dqn_agent.py        # Deep Q-Network
│   │   ├── ppo_agent.py        # Proximal Policy Optimization
│   │   └── a2c_agent.py        # Advantage Actor-Critic
│   ├── networks/               # Neural network architectures
│   │   ├── q_network.py        # CNN for DQN
│   │   ├── actor_critic.py     # Shared network for PPO/A2C
│   │   └── utils.py            # Network utilities
│   ├── environment/            # Environment wrappers
│   │   ├── wrappers.py         # Atari preprocessing
│   │   └── env_factory.py      # Environment creation
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main training loop
│   │   ├── replay_buffer.py    # Experience replay (DQN)
│   │   └── rollout_buffer.py   # Rollout storage (PPO/A2C)
│   └── utils/                  # Utilities
│       ├── logger.py           # TensorBoard logging
│       ├── checkpoint.py       # Model saving/loading
│       └── visualization.py    # Training visualization
├── scripts/                     # Command-line scripts
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── play.py                 # Watch agent play
├── notebooks/                   # Jupyter notebooks
│   ├── 01_environment_exploration.ipynb
│   ├── 02_train_dqn.ipynb
│   ├── 03_train_ppo.ipynb
│   ├── 04_train_a2c.ipynb
│   └── 05_compare_algorithms.ipynb
├── tests/                       # Unit tests
│   ├── test_agents.py
│   ├── test_networks.py
│   └── test_environment.py
├── models/                      # Saved model checkpoints
├── logs/                        # TensorBoard logs
└── docs/                        # Additional documentation
    ├── algorithms.md           # Deep dive into algorithms
    ├── architecture.md         # Network architecture details
    └── troubleshooting.md      # Common issues and solutions
```

**File Organization:**
- `models/`: Checkpoints saved as `{algorithm}_step{N}_reward{R}_{timestamp}.pt`
- `logs/`: TensorBoard logs in `runs/{algorithm}_{timestamp}/`

## Getting Started

### Google Colab Quick Start

```python
# Clone repository
!git clone https://github.com/SABRYOLA/pacman.git
%cd pacman

# Install dependencies
!pip install -r requirements.txt

# Train PPO agent (fast, good performance)
!python scripts/train.py --algorithm ppo --steps 500000

# View logs
%load_ext tensorboard
%tensorboard --logdir logs/
```

### Local Installation

**1. Clone the repository:**
```bash
git clone https://github.com/SABRYOLA/pacman.git
cd pacman
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Train an agent:**
```bash
# Train DQN
python scripts/train.py --algorithm dqn --steps 1000000

# Train PPO (recommended)
python scripts/train.py --algorithm ppo --steps 1000000

# Train A2C
python scripts/train.py --algorithm a2c --steps 1000000
```

**5. Monitor training:**
```bash
tensorboard --logdir logs/
# Open http://localhost:6006 in browser
```

**6. Evaluate trained agent:**
```bash
python scripts/evaluate.py --checkpoint models/PPO_best_reward2500.pt --algorithm ppo --episodes 100
```

**7. Watch agent play:**
```bash
python scripts/play.py --checkpoint models/PPO_best_reward2500.pt --algorithm ppo --episodes 5
```

## Expected Results

Performance benchmarks after training (approximate):

| Algorithm | 100K Steps | 500K Steps | 1M Steps | 2M Steps |
|-----------|------------|------------|----------|----------|
| DQN       | ~400       | ~1200      | ~2000    | ~3000    |
| PPO       | ~600       | ~1500      | ~2500    | ~4000    |
| A2C       | ~500       | ~1300      | ~2200    | ~3500    |

**Note:** Results vary significantly based on:
- Random seed initialization
- Hardware (GPU vs CPU affects training dynamics)
- Hyperparameter settings
- Environment stochasticity

**Typical Learning Curves:**
- **DQN**: Slow start (exploration), steady improvement after 200K steps
- **PPO**: Faster initial learning, more stable throughout training
- **A2C**: Moderate learning speed, can be less stable than PPO

## Educational Value

### Concepts Demonstrated

**Reinforcement Learning Fundamentals:**
- Value-based methods (DQN) vs Policy-based methods (PPO, A2C)
- On-policy (PPO, A2C) vs Off-policy (DQN) learning
- Exploration strategies (ε-greedy, entropy bonus)
- Credit assignment problem and solutions

**Advanced Techniques:**
- Experience replay and target networks (DQN)
- Generalized Advantage Estimation (PPO)
- Policy gradient theorem and REINFORCE
- Actor-critic architectures
- Clipped surrogate objective (PPO)

**Deep Learning for RL:**
- CNN feature extraction from visual input
- Shared vs separate networks
- Gradient clipping and optimization
- Batch normalization alternatives for RL

**Engineering Practices:**
- Modular code architecture
- Configuration management
- Experiment logging and visualization
- Model checkpointing and evaluation

### Skills Developed

**Technical Skills:**
- PyTorch neural network implementation
- RL algorithm implementation from scratch
- Environment preprocessing and wrapping
- Training loop design
- Debugging RL training issues

**ML Engineering:**
- Hyperparameter tuning
- Experiment tracking
- Code organization for ML projects
- Version control for ML code
- Reproducibility practices

## Extension Opportunities

### Beginner Extensions

1. **Hyperparameter Tuning**: Experiment with different learning rates, network sizes
2. **Reward Shaping**: Add intermediate rewards for pellet proximity
3. **Additional Metrics**: Track pellets eaten, ghost encounters
4. **Other Atari Games**: Apply algorithms to Breakout, Space Invaders, etc.

### Intermediate Extensions

1. **Dueling DQN**: Separate value and advantage streams
2. **Prioritized Experience Replay**: Sample important transitions more frequently
3. **Double DQN**: Address overestimation bias
4. **Multi-environment Training**: Parallel environments for DQN
5. **Noisy Networks**: Parameter space noise for exploration

### Advanced Extensions

1. **Rainbow DQN**: Combine all DQN improvements
2. **Curiosity-Driven Exploration (ICM)**: Intrinsic motivation
3. **Recurrent Policies**: Add LSTM/GRU for memory
4. **Distributed Training**: IMPALA, APEX-DQN
5. **Meta-Learning**: Train agents that adapt quickly to new games
6. **Neural Architecture Search**: Optimize network architecture
7. **Multi-task Learning**: Single agent for multiple Atari games

## Citation

If you use this code in your research or projects, please cite:

```bibtex
@software{pacman_rl_2024,
  author = {SABRYOLA},
  title = {Ms. Pac-Man Reinforcement Learning Project},
  year = {2024},
  url = {https://github.com/SABRYOLA/pacman}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Atari preprocessing**: Based on implementations from OpenAI Baselines and Stable-Baselines3
- **Algorithm implementations**: Inspired by original papers and open-source implementations
- **Environment**: Thanks to the Arcade Learning Environment (ALE) team

## References

**DQN:**
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.

**PPO:**
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv.

**A2C:**
- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML.

**General RL:**
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction." MIT Press.

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the repository owner.

---

**Happy Learning! 🎮🤖**