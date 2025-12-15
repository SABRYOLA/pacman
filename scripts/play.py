#!/usr/bin/env python3
"""Watch a trained agent play Ms. Pac-Man."""

import argparse
import torch
import time

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


def play_episode(agent, env, sleep_time: float = 0.03):
    """
    Play one episode with the agent.
    
    Args:
        agent: Trained RL agent
        env: Environment
        sleep_time: Time to sleep between frames
        
    Returns:
        Episode reward
    """
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    print("\nStarting episode...")
    
    while not done:
        env.render()
        
        action = agent.select_action(obs, training=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print(f"Episode finished!")
    print(f"  Reward: {episode_reward:.2f}")
    print(f"  Length: {episode_length}")
    
    return episode_reward


def main():
    parser = argparse.ArgumentParser(description="Watch trained agent play")
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
        default=5,
        help="Number of episodes to play"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.03,
        help="Sleep time between frames (seconds)"
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
    
    # Create environment with rendering
    env = create_env(args.env, seed=args.seed, render_mode="human")
    n_actions = env.action_space.n
    
    print(f"Environment: {args.env}")
    print(f"Action space: {n_actions}")
    
    # Load agent
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, args.algorithm, n_actions, device)
    print(f"Agent: {args.algorithm.upper()}")
    
    # Play episodes
    print(f"\nPlaying {args.episodes} episode(s)...")
    print("Close the window or press Ctrl+C to stop.\n")
    
    episode_rewards = []
    
    try:
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
            reward = play_episode(agent, env, args.sleep)
            episode_rewards.append(reward)
            
            if episode < args.episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)
        
        # Print summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Episodes played: {len(episode_rewards)}")
        if episode_rewards:
            import numpy as np
            print(f"Mean reward: {np.mean(episode_rewards):.2f}")
            print(f"Max reward: {np.max(episode_rewards):.2f}")
            print(f"Min reward: {np.min(episode_rewards):.2f}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
