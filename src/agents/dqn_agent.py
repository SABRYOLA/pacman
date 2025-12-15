"""Deep Q-Network (DQN) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from ..networks import QNetwork
from ..training import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Key features:
    - Experience replay to break temporal correlations
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration with decay
    """
    
    def __init__(
        self,
        n_actions: int,
        observation_shape: Tuple[int, int, int] = (4, 84, 84),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_interval: int = 1000,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        exploration_fraction: float = 0.1,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize DQN agent.
        
        Args:
            n_actions: Number of possible actions
            observation_shape: Shape of observations
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Size of replay buffer
            batch_size: Mini-batch size for training
            target_update_interval: Steps between target network updates
            exploration_initial_eps: Initial epsilon for exploration
            exploration_final_eps: Final epsilon for exploration
            exploration_fraction: Fraction of training for epsilon decay
            device: Device to run on
        """
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device
        
        # Exploration parameters
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.epsilon = exploration_initial_eps
        
        # Networks
        self.q_network = QNetwork(n_actions, observation_shape).to(device)
        self.target_network = QNetwork(n_actions, observation_shape).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, observation_shape, device)
        
        # Training statistics
        self.training_steps = 0
        self.episode_count = 0
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update_epsilon(self, progress: float) -> None:
        """
        Update exploration epsilon based on training progress.
        
        Args:
            progress: Training progress (0 to 1)
        """
        if progress < self.exploration_fraction:
            # Linear decay
            decay_progress = progress / self.exploration_fraction
            self.epsilon = self.exploration_initial_eps - (
                self.exploration_initial_eps - self.exploration_final_eps
            ) * decay_progress
        else:
            self.epsilon = self.exploration_final_eps
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.replay_buffer.add(obs, action, reward, next_obs, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value or None if buffer not ready
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Compute current Q-values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
