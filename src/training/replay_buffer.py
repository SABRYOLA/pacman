"""Experience replay buffer for DQN."""

import numpy as np
import torch
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for off-policy learning.
    
    Used by DQN to break temporal correlations and improve sample efficiency.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            observation_shape: Shape of observations (e.g., (4, 84, 84))
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory for efficiency
        self.observations = np.zeros((buffer_size, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *observation_shape), dtype=np.uint8)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        observations = torch.from_numpy(self.observations[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_observations = torch.from_numpy(self.next_observations[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)
        
        return observations, actions, rewards, next_observations, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size
