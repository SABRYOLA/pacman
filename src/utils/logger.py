"""Logging utilities for training."""

import os
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np


class Logger:
    """
    Logger for tracking training metrics and writing to TensorBoard.
    """
    
    def __init__(self, log_dir: str, algorithm: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            algorithm: Name of the algorithm (for log naming)
        """
        self.log_dir = log_dir
        self.algorithm = algorithm
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Step counters
        self.total_steps = 0
        self.num_episodes = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Name of the metric
            value: Value to log
            step: Global step (uses self.total_steps if None)
        """
        if step is None:
            step = self.total_steps
        self.writer.add_scalar(tag, value, step)
    
    def log_episode(self, reward: float, length: int) -> None:
        """
        Log episode statistics.
        
        Args:
            reward: Total episode reward
            length: Episode length
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.num_episodes += 1
        
        # Log to TensorBoard
        self.log_scalar("episode/reward", reward, self.num_episodes)
        self.log_scalar("episode/length", length, self.num_episodes)
        
        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            self.log_scalar("episode/mean_reward_100", mean_reward, self.num_episodes)
            self.log_scalar("episode/mean_length_100", mean_length, self.num_episodes)
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        for key, value in metrics.items():
            self.log_scalar(f"train/{key}", value)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_steps": self.total_steps,
            "num_episodes": self.num_episodes,
        }
        
        if self.episode_rewards:
            stats["mean_reward"] = np.mean(self.episode_rewards)
            stats["max_reward"] = np.max(self.episode_rewards)
            stats["min_reward"] = np.min(self.episode_rewards)
            stats["std_reward"] = np.std(self.episode_rewards)
        
        if self.episode_lengths:
            stats["mean_length"] = np.mean(self.episode_lengths)
        
        return stats
    
    def print_stats(self) -> None:
        """Print current statistics to console."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Steps: {stats['total_steps']:,} | Episodes: {stats['num_episodes']}")
        
        if "mean_reward" in stats:
            print(f"Reward (100ep): {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        
        if "mean_length" in stats:
            print(f"Mean Length: {stats['mean_length']:.1f}")
        
        print(f"{'='*60}\n")
    
    def close(self) -> None:
        """Close the logger."""
        self.writer.close()
