"""Advantage Actor-Critic (A2C) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
from ..networks import ActorCriticNetwork
from ..training import RolloutBuffer

# Numerical stability constant
EPSILON = 1e-8


class A2CAgent:
    """
    Advantage Actor-Critic agent with synchronous updates.
    
    Key features:
    - Synchronous policy and value updates
    - N-step returns with optional GAE
    - Entropy regularization for exploration
    - RMSprop or Adam optimizer
    """
    
    def __init__(
        self,
        n_actions: int,
        observation_shape: Tuple[int, int, int] = (4, 84, 84),
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        n_steps: int = 5,
        n_envs: int = 8,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        use_rms_prop: bool = True,
        rms_prop_eps: float = 1e-5,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize A2C agent.
        
        Args:
            n_actions: Number of possible actions
            observation_shape: Shape of observations
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda (1.0 = no GAE, just n-step returns)
            n_steps: Number of steps per environment per update
            n_envs: Number of parallel environments
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            normalize_advantage: Whether to normalize advantages
            use_rms_prop: Whether to use RMSprop (else Adam)
            rms_prop_eps: RMSprop epsilon
            device: Device to run on
        """
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.device = device
        
        # Network
        self.policy = ActorCriticNetwork(n_actions, observation_shape).to(device)
        
        # Optimizer
        if use_rms_prop:
            self.optimizer = optim.RMSprop(
                self.policy.parameters(),
                lr=learning_rate,
                eps=rms_prop_eps,
                alpha=0.99
            )
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=learning_rate,
                eps=1e-5
            )
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            n_steps, observation_shape, n_envs, gamma, gae_lambda, device
        )
        
        # Training statistics
        self.training_steps = 0
    
    def select_action(
        self,
        observations: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for multiple environments.
        
        Args:
            observations: Observations from environments (n_envs, *obs_shape)
            training: Whether in training mode
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observations).to(self.device)
            actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)
        
        return (
            actions.cpu().numpy(),
            values.cpu().numpy().flatten(),
            log_probs.cpu().numpy()
        )
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ) -> None:
        """
        Store a transition in the rollout buffer.
        
        Args:
            obs: Observations (n_envs, *obs_shape)
            action: Actions taken (n_envs,)
            reward: Rewards received (n_envs,)
            done: Episode done flags (n_envs,)
            value: Value estimates (n_envs,)
            log_prob: Log probabilities (n_envs,)
        """
        self.rollout_buffer.add(obs, action, reward, done, value, log_prob)
    
    def train_step(self, last_observations: np.ndarray) -> Dict[str, float]:
        """
        Perform A2C update.
        
        Args:
            last_observations: Final observations for bootstrap value
            
        Returns:
            Dictionary of training metrics
        """
        # Compute value for last observation
        with torch.no_grad():
            obs_tensor = torch.from_numpy(last_observations).to(self.device)
            last_value = self.policy.get_value(obs_tensor).cpu().numpy().flatten()
        
        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(last_value)
        
        # Get all data from buffer
        batch = next(self.rollout_buffer.get(batch_size=None))
        observations, actions, old_values, old_log_probs, advantages, returns = batch
        
        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPSILON)
        
        # Get current policy outputs
        _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
            observations, actions
        )
        new_values = new_values.flatten()
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss
        value_loss = ((new_values - returns) ** 2).mean()
        
        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.vf_coef * value_loss +
            self.ent_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint["training_steps"]
