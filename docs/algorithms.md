# Deep Dive into Reinforcement Learning Algorithms

This document provides detailed explanations of the three algorithms implemented in this project: DQN, PPO, and A2C.

## Table of Contents
- [Deep Q-Network (DQN)](#deep-q-network-dqn)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
- [Comparison](#comparison)

## Deep Q-Network (DQN)

### Background

DQN, introduced by DeepMind in 2015, was the first deep learning algorithm to successfully learn to play Atari games from raw pixels. It combines Q-learning with deep neural networks and introduced two key innovations that made stable learning possible.

### Core Concepts

**Q-Learning:**
The Q-value Q(s, a) represents the expected cumulative reward for taking action a in state s and following the optimal policy thereafter:

```
Q*(s, a) = E[r + γ * max_a' Q*(s', a')]
```

**Neural Network Approximation:**
Instead of maintaining a table of Q-values for each state-action pair (infeasible for high-dimensional spaces), DQN uses a neural network to approximate Q(s, a).

### Key Innovations

**1. Experience Replay**

Problem: Consecutive samples are highly correlated, leading to inefficient learning and instability.

Solution: Store experiences in a replay buffer and sample randomly during training.

```python
# Store transition
buffer.add(state, action, reward, next_state, done)

# Sample random minibatch
batch = buffer.sample(batch_size)
```

Benefits:
- Breaks temporal correlations
- Improves sample efficiency (each experience used multiple times)
- Enables off-policy learning

**2. Target Network**

Problem: Using the same network to generate targets and predictions causes instability (chasing a moving target).

Solution: Use a separate, slowly-updated target network for computing target Q-values.

```python
# Compute target Q-values with frozen target network
with torch.no_grad():
    target_q = reward + gamma * target_net(next_state).max()

# Compute loss with online network
q_value = online_net(state)[action]
loss = mse_loss(q_value, target_q)
```

Benefits:
- Reduces oscillations and divergence
- More stable learning
- Prevents feedback loops

### Algorithm Pseudocode

```
Initialize replay buffer D
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

for episode = 1 to M:
    Initialize state s₀
    
    for t = 0 to T:
        # Epsilon-greedy action selection
        if random() < ε:
            a_t = random action
        else:
            a_t = argmax_a Q(s_t, a; θ)
        
        # Execute action
        Execute a_t, observe r_t and s_{t+1}
        
        # Store transition
        Store (s_t, a_t, r_t, s_{t+1}) in D
        
        # Train
        if |D| > batch_size:
            # Sample minibatch
            Sample batch of (s, a, r, s') from D
            
            # Compute targets
            y = r + γ * max_a' Q(s', a'; θ⁻)
            
            # Update Q-network
            θ ← θ - α∇_θ(Q(s, a; θ) - y)²
        
        # Update target network
        Every C steps: θ⁻ ← θ
        
        # Decay epsilon
        ε ← max(ε_min, ε * decay)
```

### Implementation Details

**Network Architecture:**
```
Input: (batch, 4, 84, 84) grayscale frames
↓
Conv2D(32 filters, 8×8, stride 4) + ReLU
↓
Conv2D(64 filters, 4×4, stride 2) + ReLU
↓
Conv2D(64 filters, 3×3, stride 1) + ReLU
↓
Flatten
↓
FC(512) + ReLU
↓
FC(n_actions)
↓
Output: Q-values for each action
```

**Hyperparameters:**
- Learning rate: 1e-4 (Adam optimizer)
- Discount factor γ: 0.99
- Replay buffer size: 100,000
- Batch size: 32
- Target update frequency: 1,000 steps
- Initial epsilon: 1.0
- Final epsilon: 0.01
- Epsilon decay: Linear over first 10% of training

**Exploration Strategy:**
Epsilon-greedy with linear decay encourages exploration early and exploitation later.

### Pros and Cons

**Advantages:**
- Sample efficient (off-policy, reuses experiences)
- Stable with experience replay and target network
- Simple to implement and understand
- Well-suited for discrete action spaces

**Disadvantages:**
- Slower learning initially (needs to fill replay buffer)
- Memory intensive (stores many transitions)
- Cannot handle continuous actions directly
- Potentially overestimates Q-values

## Proximal Policy Optimization (PPO)

### Background

PPO, introduced by OpenAI in 2017, is a policy gradient method that achieves state-of-the-art performance while being simpler than previous methods like TRPO. It's become one of the most popular RL algorithms due to its reliability and ease of use.

### Core Concepts

**Policy Gradient:**
Instead of learning Q-values, directly optimize the policy π(a|s):

```
∇_θ J(θ) = E[∇_θ log π(a|s; θ) * A(s, a)]
```

Where A(s, a) is the advantage function (how much better is this action than average).

**Trust Region:**
Limit how much the policy can change in each update to ensure stable learning.

### Key Innovations

**1. Clipped Surrogate Objective**

Problem: Large policy updates can destabilize training.

Solution: Clip the ratio of new and old policies to limit updates.

```python
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

This prevents the policy from changing too much in a single update.

**2. Generalized Advantage Estimation (GAE)**

Problem: High variance in advantage estimates.

Solution: Use an exponentially-weighted average of n-step advantages:

```
A_GAE = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**3. Multiple Epochs**

Unlike typical policy gradients that discard data after one update, PPO reuses collected data for multiple epochs of optimization.

### Algorithm Pseudocode

```
Initialize policy network π_θ and value network V_φ

for iteration = 1 to N:
    # Collect trajectories
    for t = 1 to T:
        a_t ~ π_θ(·|s_t)
        Execute a_t, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, V(s_t), log π(a_t|s_t))
    
    # Compute advantages using GAE
    Compute advantages A_t for all t
    
    # Optimize policy
    for epoch = 1 to K:
        for minibatch in shuffle(trajectories):
            # Compute ratio
            r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
            
            # Clipped surrogate loss
            L_CLIP = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            
            # Value loss
            L_VF = (V_φ(s_t) - V_target)²
            
            # Entropy bonus
            L_S = -H(π_θ(·|s_t))
            
            # Total loss
            L = L_CLIP + c₁L_VF + c₂L_S
            
            # Update networks
            θ ← θ - α∇_θ L
```

### Implementation Details

**Network Architecture:**
```
Shared CNN backbone:
Input: (batch, 4, 84, 84)
↓
Conv2D(32, 8×8, stride 4) + ReLU
↓
Conv2D(64, 4×4, stride 2) + ReLU
↓
Conv2D(64, 3×3, stride 1) + ReLU
↓
Flatten → FC(512) + ReLU
↓
Split into two heads:
├─ Actor: FC(n_actions) → Softmax
└─ Critic: FC(1) → Value
```

**Hyperparameters:**
- Learning rate: 2.5e-4
- Discount factor γ: 0.99
- GAE lambda λ: 0.95
- Clip range ε: 0.2
- Epochs per update: 4
- Minibatch size: 256
- Steps per update: 128
- Parallel environments: 8
- Entropy coefficient: 0.01
- Value function coefficient: 0.5

### Pros and Cons

**Advantages:**
- More stable than vanilla policy gradients
- Sample efficient (reuses data multiple times)
- Works well with both continuous and discrete actions
- Easier to tune than TRPO
- Faster learning than DQN

**Disadvantages:**
- On-policy (cannot reuse old data indefinitely)
- Requires multiple parallel environments for efficiency
- More complex than DQN
- Sensitive to hyperparameters

## Advantage Actor-Critic (A2C)

### Background

A2C is a synchronous version of A3C (Asynchronous Advantage Actor-Critic), introduced by DeepMind in 2016. It combines policy gradients (actor) with value function approximation (critic).

### Core Concepts

**Actor-Critic Framework:**
- **Actor**: The policy π(a|s) that selects actions
- **Critic**: The value function V(s) that evaluates states
- **Advantage**: A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)

The actor is trained using the advantage estimated by the critic.

### Key Features

**1. Synchronous Updates**

Unlike A3C's asynchronous workers, A2C waits for all parallel environments to complete their steps before updating. This is simpler and often more efficient on modern hardware.

**2. N-step Returns**

Instead of one-step TD learning, use n-step returns for better credit assignment:

```
R_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γⁿV(s_{t+n})
```

**3. Entropy Regularization**

Add an entropy bonus to encourage exploration:

```
L = L_policy + c₁L_value - c₂H(π)
```

Higher entropy = more random policy = more exploration.

### Algorithm Pseudocode

```
Initialize policy π_θ and value V_φ

for iteration = 1 to N:
    # Collect n-step rollouts from parallel environments
    for t = 1 to n_steps:
        for env in parallel_envs:
            a ~ π_θ(·|s)
            Execute a, observe r, s'
            Store (s, a, r, V(s))
    
    # Compute n-step returns and advantages
    for t in reverse(1 to n_steps):
        R_t = r_t + γR_{t+1}
        A_t = R_t - V(s_t)
    
    # Update networks
    # Policy loss (actor)
    L_π = -log π_θ(a|s) * A
    
    # Value loss (critic)
    L_V = (V_φ(s) - R)²
    
    # Entropy loss
    L_H = -H(π_θ(·|s))
    
    # Total loss
    L = L_π + c₁L_V + c₂L_H
    
    # Update
    θ, φ ← θ, φ - α∇L
```

### Implementation Details

**Network Architecture:**
Same as PPO (shared CNN backbone with actor/critic heads)

**Hyperparameters:**
- Learning rate: 7e-4
- Discount factor γ: 0.99
- N-steps: 5
- Parallel environments: 8
- Entropy coefficient: 0.01
- Value function coefficient: 0.5
- Optimizer: RMSprop (traditionally, though Adam also works)

### Pros and Cons

**Advantages:**
- Simpler than PPO (no clipping, single update per batch)
- Faster updates than PPO (fewer epochs)
- Lower variance than vanilla policy gradients
- Works with both continuous and discrete actions

**Disadvantages:**
- Less stable than PPO
- On-policy (cannot reuse old data)
- Can be sensitive to hyperparameters
- May require more tuning than PPO

## Comparison

### Performance Characteristics

| Algorithm | Sample Efficiency | Stability | Speed | Ease of Tuning |
|-----------|------------------|-----------|-------|----------------|
| DQN       | High            | High      | Slow  | Easy           |
| PPO       | Medium          | High      | Fast  | Medium         |
| A2C       | Medium          | Medium    | Fast  | Hard           |

### When to Use Each

**Use DQN when:**
- Sample efficiency is critical
- You have discrete actions
- You can afford memory for replay buffer
- Stability is more important than speed

**Use PPO when:**
- You want reliable, robust performance
- You have both discrete or continuous actions
- You can run parallel environments
- You want state-of-the-art results

**Use A2C when:**
- You want fast training
- You prefer simplicity over stability
- You're familiar with tuning actor-critic methods
- You're building on top of the algorithm

### Theoretical Comparison

**On-policy vs Off-policy:**
- DQN: Off-policy (can reuse old data)
- PPO/A2C: On-policy (limited data reuse)

**Value-based vs Policy-based:**
- DQN: Learns Q-values, derives policy
- PPO/A2C: Directly learns policy

**Variance vs Bias Trade-off:**
- DQN: Lower variance, higher bias
- PPO: Balanced
- A2C: Higher variance, lower bias

### Practical Recommendations

1. **Start with PPO**: Most reliable for general use
2. **Try DQN**: If sample efficiency is critical or actions are discrete
3. **Experiment with A2C**: If you need speed and are willing to tune

## Further Reading

**DQN:**
- Original paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Improvements: Double DQN, Dueling DQN, Prioritized Experience Replay, Rainbow DQN

**PPO:**
- Original paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- Predecessor: TRPO (Trust Region Policy Optimization)

**A2C:**
- Original paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- Related: A3C (asynchronous version), GAE (Generalized Advantage Estimation)

**General RL:**
- Textbook: "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)
- Course: CS285 Deep RL (UC Berkeley)
