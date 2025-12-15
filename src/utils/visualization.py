"""Visualization utilities for training results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os


def plot_training_results(
    rewards: List[float],
    losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training results.
    
    Args:
        rewards: List of episode rewards
        losses: Optional list of training losses
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2 if losses else 1, 1, figsize=(12, 8 if losses else 6))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, label="Episode Reward")
    
    # Plot moving average
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0].plot(range(99, len(rewards)), moving_avg, label="100-Episode Average", linewidth=2)
    
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses:
        axes[1].plot(losses, alpha=0.5)
        axes[1].set_xlabel("Update Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of multiple algorithms.
    
    Args:
        results: Dictionary mapping algorithm names to reward lists
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(14, 6))
    
    for algorithm, rewards in results.items():
        # Plot moving average
        if len(rewards) >= 100:
            moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
            plt.plot(range(99, len(rewards)), moving_avg, label=algorithm, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (100 episodes)")
    plt.title("Algorithm Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution(
    data: List[float],
    title: str = "Distribution",
    xlabel: str = "Value",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot distribution of values.
    
    Args:
        data: List of values
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(data), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    plt.axvline(np.median(data), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.2f}')
    
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
