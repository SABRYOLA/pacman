# Network Architecture Details

This document provides detailed information about the neural network architectures used in this project.

## Overview

All algorithms use convolutional neural networks (CNNs) to process visual input from the Atari environment. The architectures are based on the original DQN paper with some modifications.

## Common Design Principles

### Why CNNs?

**Visual Input Processing:**
- Atari games provide visual observations (images)
- CNNs excel at extracting spatial features from images
- Hierarchical feature learning: edges → shapes → objects

**Translation Invariance:**
- Game objects can appear anywhere on screen
- CNNs learn features that work regardless of position
- Shared weights reduce parameters

**Computational Efficiency:**
- Much fewer parameters than fully connected networks
- Local connectivity reduces computation
- Parameter sharing across spatial locations

### Input Preprocessing

**From Raw Pixels to Network Input:**

1. **Original Frame**: 210×160×3 RGB image (100,800 values)
2. **Grayscale Conversion**: 210×160×1 (33,600 values, 67% reduction)
3. **Resize**: 84×84×1 (7,056 values, 93% total reduction)
4. **Frame Stacking**: 84×84×4 (28,224 values but with temporal info)

**Why These Transformations?**
- **Grayscale**: Color not crucial for most Atari games
- **Downsampling**: Reduces computation while keeping important features
- **Frame Stacking**: Provides motion information (velocity of objects)

## Q-Network Architecture (DQN)

### Network Diagram

```
Input: (batch, 4, 84, 84) [4 stacked grayscale frames]
  ↓
Conv Layer 1: 32 filters, 8×8 kernel, stride 4, ReLU
  Output: (batch, 32, 20, 20)
  Receptive field: 8×8
  ↓
Conv Layer 2: 64 filters, 4×4 kernel, stride 2, ReLU
  Output: (batch, 64, 9, 9)
  Receptive field: 28×28
  ↓
Conv Layer 3: 64 filters, 3×3 kernel, stride 1, ReLU
  Output: (batch, 64, 7, 7)
  Receptive field: 52×52
  ↓
Flatten: (batch, 3136)
  ↓
Fully Connected 1: 512 units, ReLU
  Output: (batch, 512)
  ↓
Fully Connected 2: n_actions units (no activation)
  Output: (batch, n_actions) [Q-values]
```

### Design Rationale

**Convolutional Layers:**
- **Layer 1** (8×8, stride 4): Aggressive downsampling, captures large features
- **Layer 2** (4×4, stride 2): Medium-scale features (objects, shapes)
- **Layer 3** (3×3, stride 1): Fine details and spatial relationships

**Increasing Filters:**
32 → 64 → 64 allows the network to learn increasingly complex features

**Fully Connected Layers:**
- Combine spatial features into abstract representations
- 512 units provide sufficient capacity without overfitting

**Output Layer:**
- Linear activation (no softmax) because Q-values can be any real number
- One output per action

### Parameter Count

```
Conv1: 32 × (4 × 8 × 8 + 1) = 8,224
Conv2: 64 × (32 × 4 × 4 + 1) = 32,832
Conv3: 64 × (64 × 3 × 3 + 1) = 36,928
FC1:   512 × (3136 + 1) = 1,606,144
FC2:   n_actions × (512 + 1) ≈ 4,617 (for 9 actions)

Total: ~1.7M parameters
```

## Actor-Critic Architecture (PPO & A2C)

### Network Diagram

```
Input: (batch, 4, 84, 84)
  ↓
┌─────────────────────────┐
│  Shared CNN Backbone    │
│                         │
│  Conv1: 32, 8×8, s=4   │
│  Conv2: 64, 4×4, s=2   │
│  Conv3: 64, 3×3, s=1   │
│  Flatten                │
│  FC: 512 units          │
└─────────────────────────┘
  ↓
  Features: (batch, 512)
  ↓
  ├──────────────────┬──────────────────┐
  ↓                  ↓                  ↓
Actor Head       Critic Head
FC(n_actions)    FC(1)
Softmax          Linear
  ↓                  ↓
Action Probs     Value Estimate
(batch, n_act)   (batch, 1)
```

### Design Rationale

**Shared Backbone:**
- Same visual features useful for both policy and value
- Reduces parameters and computation
- Enables faster learning through shared representations

**Separate Heads:**
- Policy and value have different output ranges and interpretations
- Separate heads allow specialization
- Can be trained with different loss weights

**Actor Head:**
- Outputs logits for categorical distribution
- Softmax converts to probabilities
- Sampling from distribution provides stochastic policy

**Critic Head:**
- Single output: scalar value estimate
- Linear activation (values can be any real number)
- Trained with MSE loss

### Parameter Count

```
Shared Backbone: ~1.6M (same as DQN up to FC layer)
Actor Head: n_actions × (512 + 1) ≈ 4,617
Critic Head: 1 × (512 + 1) = 513

Total: ~1.6M parameters
```

## Implementation Details

### Weight Initialization

**Orthogonal Initialization:**
```python
nn.init.orthogonal_(layer.weight, gain=1.0)
nn.init.constant_(layer.bias, 0.0)
```

**Why Orthogonal?**
- Preserves gradient norms during backprop
- Helps prevent vanishing/exploding gradients
- Empirically works well for RL

**Gain Values:**
- 1.0 for most layers
- √2 for layers before ReLU (preserves variance)

### Activation Functions

**ReLU (Rectified Linear Unit):**
```python
f(x) = max(0, x)
```

**Advantages:**
- No vanishing gradient problem
- Computationally efficient
- Sparse activation (many zeros)
- Standard choice for RL

**Alternatives Not Used:**
- Sigmoid/Tanh: Can cause vanishing gradients
- Leaky ReLU: No significant benefit observed
- GELU/Swish: More computation, minimal gain for RL

### Normalization

**Why No Batch Normalization?**

Batch normalization is typically avoided in RL because:
- Breaks independence of samples (RL already has correlation issues)
- Adds complexity during inference
- Moving statistics can interfere with target networks (DQN)
- Not necessary with proper initialization and learning rates

**Input Normalization:**
```python
x = x.float() / 255.0  # Scale to [0, 1]
```

Simple division is sufficient for Atari frames.

## Computational Complexity

### Forward Pass Complexity

**Q-Network:**
```
Conv1: 4 × 32 × 8 × 8 × 20 × 20 = 1.6M ops
Conv2: 32 × 64 × 4 × 4 × 9 × 9 = 6.6M ops
Conv3: 64 × 64 × 3 × 3 × 7 × 7 = 1.8M ops
FC1:   3136 × 512 = 1.6M ops
FC2:   512 × n_actions ≈ 4.6K ops

Total: ~11.6M FLOPs per sample
```

**Actor-Critic:**
Similar to Q-Network plus small overhead for dual heads (~12M FLOPs)

### Memory Requirements

**Model Weights:**
- DQN: 1.7M params × 4 bytes = 6.8 MB
- Two networks (online + target): 13.6 MB
- Actor-Critic: 6.4 MB

**Activations (batch size 32):**
- Input: 32 × 4 × 84 × 84 × 1 byte = 0.9 MB
- Feature maps: ~5 MB
- Total: ~6 MB per forward pass

**Replay Buffer (DQN):**
- 100K transitions × (2 frames × 28KB + small metadata) = ~5.6 GB
- Largest memory consumer!

## Architecture Variations

### Deeper Networks

**Why Not Deeper?**
- Atari images are relatively simple
- More depth = more parameters = longer training
- Diminishing returns for visual complexity of Atari games

**When to Add Depth:**
- More complex visual environments
- If underfitting (low training performance)
- Modern games with rich graphics

### Larger Networks

**Scaling Width:**
- 2x filters: 4x parameters, better capacity
- Useful for complex environments
- Requires more data to train

**Scaling FC Layers:**
- 512 → 1024 units: doubles FC parameters
- May help with complex state representations

### Attention Mechanisms

**Not Used Because:**
- Overkill for Atari
- Adds significant complexity
- No clear benefit demonstrated

**When to Consider:**
- Partial observability
- Very large state spaces
- Need to focus on specific regions

## Optimization Details

### Learning Rates

**DQN: 1e-4**
- Conservative for off-policy learning
- Adam optimizer with default β values

**PPO: 2.5e-4**
- Slightly higher (on-policy can tolerate)
- Clipping provides additional stability

**A2C: 7e-4**
- Highest learning rate
- RMSprop with α=0.99

### Gradient Clipping

All algorithms use gradient norm clipping:
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)  # DQN
torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)   # PPO/A2C
```

**Purpose:**
- Prevents exploding gradients
- Stabilizes training
- Essential for RL (high variance gradients)

## Performance Optimizations

### Batched Processing

**Vectorized Environments (PPO/A2C):**
- Process 8 environments in parallel
- Efficient GPU utilization
- 8x throughput improvement

**Mini-batch Training:**
- Process 32-256 samples at once
- Amortize kernel launch overhead
- Better GPU memory bandwidth utilization

### Mixed Precision Training

**Not Implemented But Possible:**
```python
from torch.cuda.amp import autocast, GradScaler

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

**Benefits:**
- 2x faster training
- Reduced memory usage
- Maintained accuracy (for most tasks)

## Debugging and Monitoring

### Network Health Checks

**During Training, Monitor:**
- Gradient norms (should be 0.1-10)
- Weight norms (should grow slowly)
- Activation statistics (avoid dead ReLUs)
- Loss magnitude (should decrease)

**Warning Signs:**
- NaN/Inf values → numerical instability
- Zero gradients → dead neurons
- Exploding gradients → reduce learning rate
- No improvement → architecture may be too small

### Visualization

**Useful Visualizations:**
- Feature map visualization (what network sees)
- Gradient flow (which layers learning)
- Q-value distributions (DQN)
- Policy entropy (PPO/A2C)

## Summary

The architectures used in this project are proven designs based on extensive RL research. They balance:
- **Capacity**: Enough parameters to learn complex policies
- **Efficiency**: Fast inference and training
- **Stability**: Robust to hyperparameter choices

For most modifications, start with the base architecture and only add complexity if needed for your specific problem.
