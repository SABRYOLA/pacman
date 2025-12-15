#!/usr/bin/env python3
"""Evaluation script for trained RL agents."""

import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.agents import DQNAgent, PPOAgent, A2CAgent
from src.environment import create_env
from src.networks import get_device


def load_agent(checkpoint_path: str, algorithm: str, n_actions: int, device: torch.device):
    """Load trained agent from checkpoint."""
    if algorithm == "dqn":
        agent = DQNAgent(n_actions=n_actions, device=device)
        agent.load(checkpoint_path)
    elif algorithm == "ppo":
        agent = PPOAgent(n_actions=n_actions, n_envs=1, device=device)
        agent.load(checkpoint_path)
    elif algorithm == "a2c":
        agent = A2CAgent(n_actions=n_actions, n_envs=1, device=device)
        agent.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def evaluate_agent(agent, env, n_episodes: int = 100, render: bool = False):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained RL agent
        env: Environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render environment
        
    Returns:
        Dictionary of evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dqn", "ppo", "a2c"],
        required=True,
        help="RL algorithm"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/MsPacman-v5",
        help="Environment name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    render_mode = "human" if args.render else None
    env = create_env(args.env, seed=args.seed, render_mode=render_mode)
    n_actions = env.action_space.n
    
    print(f"Environment: {args.env}")
    print(f"Action space: {n_actions}")
    
    # Load agent
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, args.algorithm, n_actions, device)
    print(f"Agent: {args.algorithm.upper()}")
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    stats = evaluate_agent(agent, env, args.episodes, args.render)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print("="*60 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()
