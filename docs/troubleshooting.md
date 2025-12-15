# Troubleshooting Guide

This guide helps you diagnose and fix common issues when training RL agents.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Performance Issues](#performance-issues)
- [Environment Issues](#environment-issues)
- [Hardware and System Issues](#hardware-and-system-issues)

## Installation Issues

### Problem: ImportError for gymnasium

**Error:**
```
ImportError: No module named 'gymnasium'
```

**Solution:**
```bash
pip install gymnasium[atari]
pip install ale-py
```

**Note:** Make sure you're using `gymnasium` (not the old `gym`). Gymnasium is the maintained successor to OpenAI Gym.

---

### Problem: ROM files not found

**Error:**
```
ale_py.roms.utils.RomNotFound: ROM not found for ALE/MsPacman-v5
```

**Solution:**
```bash
pip install "gymnasium[atari, accept-rom-license]"
```

The `accept-rom-license` option automatically downloads the ROMs (requires accepting Atari 2600 ROM license).

---

### Problem: CUDA/PyTorch version mismatch

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
Check CUDA version:
```bash
nvidia-smi  # Check CUDA version
```

Reinstall PyTorch with correct CUDA version:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Problem: OpenCV import error

**Error:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

**Alternative:**
```bash
pip install opencv-python-headless
```

## Training Issues

### Problem: Agent not learning (reward stays constant)

**Symptoms:**
- Reward doesn't improve after thousands of steps
- Agent takes random-looking actions
- Loss decreases but performance doesn't improve

**Possible Causes and Solutions:**

**1. Learning rate too high/low:**
```yaml
# Try adjusting learning rate
learning_rate: 0.0001  # DQN
learning_rate: 0.00025 # PPO
learning_rate: 0.0007  # A2C
```

**2. Not enough exploration (DQN):**
```yaml
# Increase exploration
exploration_fraction: 0.2  # Explore for longer
exploration_final_eps: 0.05  # Keep some exploration
```

**3. Reward clipping issues:**
Check if rewards are being clipped appropriately. For sparse reward environments, clipping might hide the signal.

**4. Network too small/large:**
- Too small: Can't represent complex policies
- Too large: Takes forever to learn, may overfit

**5. Check if learning has started:**
```python
# DQN needs replay buffer to fill before learning
learning_starts: 50000  # Try reducing if testing
```

---

### Problem: Training is unstable (reward fluctuates wildly)

**Symptoms:**
- Reward goes up then crashes down
- Loss spikes randomly
- Agent "forgets" good behavior

**Solutions:**

**1. Reduce learning rate:**
```yaml
learning_rate: 0.00005  # Half the default
```

**2. Increase batch size (PPO/A2C):**
```yaml
batch_size: 512  # More stable gradients
```

**3. Enable gradient clipping (should already be on):**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**4. Reduce PPO clip range:**
```yaml
clip_range: 0.1  # More conservative updates
```

**5. Use more environments (PPO/A2C):**
```yaml
n_envs: 16  # More diverse experiences
```

---

### Problem: Loss becomes NaN or Inf

**Error:**
```
RuntimeError: Loss is NaN
```

**Causes and Solutions:**

**1. Exploding gradients:**
```yaml
# Reduce learning rate
learning_rate: 0.00001

# Stronger gradient clipping
max_grad_norm: 0.1
```

**2. Numerical instability:**
```python
# Add epsilon to log computations
log_prob = torch.log(prob + 1e-8)

# Use log-sum-exp trick for stability
```

**3. Bad initialization:**
```python
# Check weight initialization
# Make sure init_weights is being called
```

**4. Reward scale too large:**
```python
# Normalize rewards
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

---

### Problem: DQN target network not updating

**Symptoms:**
- Loss stays constant
- Q-values don't change
- No learning after initial period

**Solution:**
```python
# Check target update frequency
target_update_interval: 1000  # Update every 1000 steps

# Verify update is happening (add logging)
if step % target_update_interval == 0:
    print(f"Updating target network at step {step}")
    target_network.load_state_dict(q_network.state_dict())
```

---

### Problem: PPO policy updates too large

**Symptoms:**
- Performance degrades after good episodes
- Policy ratio warnings in logs
- Unstable training

**Solution:**
```yaml
# Reduce clip range
clip_range: 0.1

# Reduce learning rate
learning_rate: 0.0001

# Fewer optimization epochs
n_epochs: 2
```

---

### Problem: A2C has high variance

**Symptoms:**
- Reward very noisy
- Hard to see learning trend
- Unstable value estimates

**Solutions:**

**1. Increase n_envs:**
```yaml
n_envs: 16  # More parallel environments
```

**2. Use GAE:**
```yaml
gae_lambda: 0.95  # Instead of 1.0
```

**3. Normalize advantages:**
```yaml
normalize_advantage: true  # Should already be enabled
```

**4. Increase value function coefficient:**
```yaml
vf_coef: 1.0  # Stronger value learning
```

## Performance Issues

### Problem: Training is very slow

**Symptoms:**
- Takes days instead of hours
- Low GPU utilization
- High CPU usage

**Solutions:**

**1. Check device:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU
```

**2. Increase batch size (if memory allows):**
```yaml
batch_size: 64  # or 128, 256
```

**3. Reduce evaluation frequency:**
```yaml
eval_freq: 50000  # Less frequent evaluation
eval_episodes: 5  # Fewer episodes per eval
```

**4. Use more environments (PPO/A2C):**
```yaml
n_envs: 16  # More parallel processing
```

**5. Optimize environment:**
```python
# Disable rendering during training
render_mode: null
```

---

### Problem: Out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**1. Reduce batch size:**
```yaml
batch_size: 16  # Smaller batches
```

**2. Reduce number of environments:**
```yaml
n_envs: 4  # Fewer parallel environments
```

**3. Reduce buffer size (DQN):**
```yaml
buffer_size: 50000  # Half the default
```

**4. Use CPU for buffer storage:**
```python
# Modify replay buffer to use CPU
device = "cpu"  # For buffer, use GPU only for training
```

**5. Clear cache periodically:**
```python
import torch
torch.cuda.empty_cache()
```

---

### Problem: High CPU usage even with GPU

**Cause:**
Environment simulation happens on CPU

**Solutions:**

**1. Reduce environment complexity:**
```python
# Skip more frames
frame_skip: 8  # Default is 4
```

**2. Use compiled environments:**
```bash
# Install optimized ALE
pip install ale-py[all]
```

**3. Optimize data loading:**
```python
# Use proper multiprocessing
num_workers = 4  # For vectorized envs
```

## Environment Issues

### Problem: Environment doesn't reset properly

**Symptoms:**
- Episode never ends
- Weird observations after reset
- Rewards don't make sense

**Solution:**
```python
# Make sure to use the new reset API
obs, info = env.reset(seed=42)  # Returns tuple

# Old API (don't use):
# obs = env.reset()  # Only returns obs
```

---

### Problem: Frame stacking not working

**Symptoms:**
- Agent doesn't learn temporal patterns
- Can't avoid moving obstacles

**Check:**
```python
# Verify frame stack shape
print(obs.shape)  # Should be (4, 84, 84)

# Make sure FrameStack wrapper is applied
from src.environment import make_atari_env
env = make_atari_env("ALE/MsPacman-v5", frame_stack=4)
```

---

### Problem: Reward too sparse

**Symptoms:**
- Agent never gets positive reward
- Random policy performs better
- No learning signal

**Solutions:**

**1. Reward shaping (careful!):**
```python
# Add intermediate rewards
reward = original_reward
if pellet_distance_decreased:
    reward += 0.01  # Small bonus
```

**2. Reduce epsilon faster:**
```yaml
exploration_fraction: 0.05  # Exploit more after learning starts
```

**3. Use curriculum learning:**
```python
# Start with easier game configurations
# Gradually increase difficulty
```

---

### Problem: Observation space mismatch

**Error:**
```
RuntimeError: Expected input shape (4, 84, 84) but got (84, 84, 4)
```

**Solution:**
```python
# Check observation shape and transpose if needed
if obs.shape == (84, 84, 4):
    obs = obs.transpose(2, 0, 1)  # HWC -> CHW
```

## Hardware and System Issues

### Problem: GPU not being used

**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
```

**Solutions:**

**1. Verify CUDA installation:**
```bash
nvidia-smi  # Should show GPU
nvcc --version  # Should show CUDA version
```

**2. Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**3. Explicitly set device:**
```bash
python scripts/train.py --algorithm ppo --device cuda
```

---

### Problem: Disk space running out

**Cause:**
Large TensorBoard logs and model checkpoints

**Solutions:**

**1. Reduce save frequency:**
```yaml
save_freq: 100000  # Save less often
```

**2. Clean old checkpoints:**
```bash
# Keep only best models
rm models/*step*.pt
# Keep best models only
```

**3. Disable TensorBoard temporarily:**
```python
# Comment out logger if needed
# logger.log_scalar(...)
```

---

### Problem: Can't visualize training (rendering)

**Error:**
```
gym.error.DependencyNotInstalled: pygame is not installed
```

**Solution:**
```bash
pip install pygame

# For headless servers
export SDL_VIDEODRIVER=dummy
```

**Alternative - Use TensorBoard:**
```bash
tensorboard --logdir logs/
```

## Common Error Messages

### "Replay buffer not full"

**Meaning:** DQN hasn't collected enough experiences

**Solution:** Wait or reduce `learning_starts`:
```yaml
learning_starts: 10000  # Start learning earlier
```

---

### "Buffer overflow"

**Meaning:** Trying to add to rollout buffer that's full

**Solution:** This is a bug. Make sure `buffer.reset()` is called after training.

---

### "Dimension mismatch"

**Meaning:** Input/output shapes don't match

**Debug:**
```python
print(f"Input shape: {input.shape}")
print(f"Expected shape: {expected_shape}")
print(f"Network output shape: {output.shape}")
```

---

### "Gradient underflow"

**Meaning:** Gradients too small (vanishing gradients)

**Solutions:**
- Increase learning rate
- Check network architecture
- Verify weight initialization

## Debugging Checklist

When things go wrong:

- [ ] Check logs for errors/warnings
- [ ] Verify GPU is being used
- [ ] Check that loss is decreasing
- [ ] Verify observations look correct
- [ ] Check reward scale (not too large/small)
- [ ] Monitor gradient norms
- [ ] Compare with baseline hyperparameters
- [ ] Try reducing problem complexity
- [ ] Test on simpler environment first
- [ ] Check for NaN/Inf in outputs

## Getting Help

If you're still stuck:

1. **Check logs:** Look for error messages and warnings
2. **Search issues:** GitHub issues for similar problems
3. **Simplify:** Reduce to minimal failing example
4. **Share:** Post issue with:
   - Error message
   - Configuration used
   - Environment details
   - Steps to reproduce

## Additional Resources

- **Spinning Up in Deep RL**: https://spinningup.openai.com/
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **RL Discord Communities**: Various RL communities for help
- **Papers With Code**: Find SOTA implementations and benchmarks
