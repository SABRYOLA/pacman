"""Proximal Policy Optimization (PPO) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict
from ..networks import ActorCriticNetwork
from ..training import RolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization agent with clipped surrogate objective.
    
    Key features:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs of mini-batch updates
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        n_actions: int,
        observation_shape: Tuple[int, int, int] = (4, 84, 84),
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        n_epochs: int = 4,
        n_steps: int = 128,
        n_envs: int = 8,
        batch_size: int = 256,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize PPO agent.
        
        Args:
            n_actions: Number of possible actions
            observation_shape: Shape of observations
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: Clipping parameter for policy loss
            clip_range_vf: Clipping parameter for value loss (None = no clipping)
            n_epochs: Number of epochs per update
            n_steps: Steps per environment per update
            n_envs: Number of parallel environments
            batch_size: Mini-batch size
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            normalize_advantage: Whether to normalize advantages
            device: Device to run on
        """
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.device = device
        
        # Network
        self.policy = ActorCriticNetwork(n_actions, observation_shape).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        
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
        Perform PPO update.
        
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
        
        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Multiple epochs
        for epoch in range(self.n_epochs):
            # Generate mini-batches
            for batch in self.rollout_buffer.get(self.batch_size):
                observations, actions, old_values, old_log_probs, advantages, returns = batch
                
                # Normalize advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    observations, actions
                )
                new_values = new_values.flatten()
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss with optional clipping
                if self.clip_range_vf is not None:
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss_1 = (new_values - returns) ** 2
                    value_loss_2 = (value_pred_clipped - returns) ** 2
                    value_loss = torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = ((new_values - returns) ** 2).mean()
                
                # Entropy loss
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
                
                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.training_steps += 1
        
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Return metrics
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
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
