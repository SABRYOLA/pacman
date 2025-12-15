"""Rollout buffer for on-policy algorithms (PPO, A2C)."""

import numpy as np
import torch
from typing import Tuple, Optional, Generator


class RolloutBuffer:
    """
    Buffer for storing trajectories experienced during on-policy training.
    
    Used by PPO and A2C to collect experience and compute advantages.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Number of steps per environment
            observation_shape: Shape of observations (e.g., (4, 84, 84))
            n_envs: Number of parallel environments
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Pre-allocate memory
        self.observations = np.zeros((buffer_size, n_envs, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        # Computed during GAE
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ) -> None:
        """
        Add a step of experience to the buffer.
        
        Args:
            obs: Observations (n_envs, *observation_shape)
            action: Actions taken (n_envs,)
            reward: Rewards received (n_envs,)
            done: Episode done flags (n_envs,)
            value: Value estimates (n_envs,)
            log_prob: Log probabilities of actions (n_envs,)
        """
        if self.ptr >= self.buffer_size:
            raise RuntimeError("Buffer overflow - call reset() before adding more data")
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: np.ndarray) -> None:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value estimate for the last observation
        """
        last_gae_lam = 0
        
        for step in range(self.buffer_size - 1, -1, -1):
            if step == self.buffer_size - 1:
                next_value = last_value
            else:
                next_value = self.values[step + 1]
            
            next_non_terminal = 1.0 - self.dones[step]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns are advantages + values
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None) -> Generator:
        """
        Generate random mini-batches from the buffer.
        
        Args:
            batch_size: Size of mini-batches (if None, return all data)
            
        Yields:
            Tuple of (observations, actions, values, log_probs, advantages, returns)
        """
        if not self.full:
            raise RuntimeError("Buffer not full - cannot sample yet")
        
        # Flatten batch dimension
        observations = self.observations.reshape(-1, *self.observation_shape)
        actions = self.actions.reshape(-1)
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        
        total_size = self.buffer_size * self.n_envs
        indices = np.arange(total_size)
        
        if batch_size is None:
            batch_size = total_size
        
        # Generate random batches
        np.random.shuffle(indices)
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                torch.from_numpy(observations[batch_indices]).to(self.device),
                torch.from_numpy(actions[batch_indices]).to(self.device),
                torch.from_numpy(values[batch_indices]).to(self.device),
                torch.from_numpy(log_probs[batch_indices]).to(self.device),
                torch.from_numpy(advantages[batch_indices]).to(self.device),
                torch.from_numpy(returns[batch_indices]).to(self.device)
            )
    
    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0
        self.full = False
