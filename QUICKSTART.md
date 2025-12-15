# Quick Start Guide

Get started with Ms. Pac-Man RL in under 5 minutes!

## Installation (Local)

```bash
# Clone repository
git clone https://github.com/SABRYOLA/pacman.git
cd pacman

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Installation (Google Colab)

```python
!git clone https://github.com/SABRYOLA/pacman.git
%cd pacman
!pip install -r requirements.txt
```

## Train Your First Agent (PPO - Recommended)

```bash
# Train for 100K steps (~30 mins on GPU)
python scripts/train.py --algorithm ppo --steps 100000

# Monitor training in real-time
tensorboard --logdir logs/
```

## View Training Progress

Open browser to http://localhost:6006 to see:
- Episode rewards over time
- Training loss curves
- Policy entropy
- Value function estimates

## Evaluate Trained Agent

```bash
# Run 100 evaluation episodes
python scripts/evaluate.py \
  --checkpoint models/PPO_best_reward2000.pt \
  --algorithm ppo \
  --episodes 100
```

## Watch Agent Play

```bash
# Watch 5 episodes with rendering
python scripts/play.py \
  --checkpoint models/PPO_best_reward2000.pt \
  --algorithm ppo \
  --episodes 5
```

## Train All Three Algorithms

```bash
# DQN (1M steps, ~3 hours GPU)
python scripts/train.py --algorithm dqn --steps 1000000

# PPO (1M steps, ~2 hours GPU)
python scripts/train.py --algorithm ppo --steps 1000000

# A2C (1M steps, ~2 hours GPU)
python scripts/train.py --algorithm a2c --steps 1000000
```

## Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/ directory and run:
# 1. 01_environment_exploration.ipynb - Understand the environment
# 2. 02_train_dqn.ipynb - Train DQN agent
# 3. 03_train_ppo.ipynb - Train PPO agent
# 4. 04_train_a2c.ipynb - Train A2C agent
# 5. 05_compare_algorithms.ipynb - Compare results
```

## Common Commands

### Custom Training Parameters
```bash
# Custom learning rate and steps
python scripts/train.py \
  --algorithm ppo \
  --steps 500000 \
  --config configs/ppo_config.yaml

# Use CPU instead of GPU
python scripts/train.py \
  --algorithm dqn \
  --device cpu \
  --steps 100000
```

### Debugging
```bash
# Check imports work
python -c "from src.agents import DQNAgent, PPOAgent, A2CAgent; print('✓ OK')"

# Verify GPU available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

## Expected Performance

After training for 1M steps, you should see approximately:

| Algorithm | Average Score | Time (GPU) |
|-----------|---------------|------------|
| DQN       | 2000-3000     | 2-4 hours  |
| PPO       | 2500-4000     | 2-3 hours  |
| A2C       | 2200-3500     | 2-3 hours  |

## Troubleshooting

**Problem: CUDA out of memory**
```bash
# Reduce batch size or number of environments
python scripts/train.py --algorithm ppo --steps 100000
# Then edit configs/ppo_config.yaml: 
#   n_envs: 4 (instead of 8)
#   batch_size: 128 (instead of 256)
```

**Problem: Training too slow**
```bash
# Use fewer evaluation episodes
# Edit config files: eval_episodes: 5 (instead of 10)
```

**Problem: Agent not learning**
- Check TensorBoard - loss should decrease
- Ensure GPU is being used (`--device cuda`)
- Try default hyperparameters first
- Read docs/troubleshooting.md

## Next Steps

1. **Learn**: Read docs/algorithms.md for algorithm details
2. **Experiment**: Try different hyperparameters
3. **Extend**: Apply to other Atari games
4. **Contribute**: Implement Dueling DQN, Rainbow, etc.

## Resources

- **Documentation**: See README.md and docs/
- **Notebooks**: Interactive tutorials in notebooks/
- **Tests**: Examples in tests/
- **Issues**: https://github.com/SABRYOLA/pacman/issues

## Quick Tips

💡 **Start with PPO** - Most reliable and fastest to good performance
💡 **Use TensorBoard** - Essential for monitoring training
💡 **Save frequently** - Training can crash, checkpoints are important
💡 **Reduce steps first** - Test with 10K-100K before full 1M+ run
💡 **Read logs** - They tell you what's happening

Happy training! 🎮🤖
