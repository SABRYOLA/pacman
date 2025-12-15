#!/usr/bin/env python3
"""Training script for Ms. Pac-Man RL agents."""

import argparse
import yaml
import torch
import os
from datetime import datetime

from src.agents import DQNAgent, PPOAgent, A2CAgent
from src.environment import create_env, create_vec_env
from src.training import DQNTrainer, OnPolicyTrainer
from src.utils import Logger, CheckpointManager
from src.networks import get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(algorithm: str, config: dict, n_actions: int, device: torch.device):
    """Create agent based on algorithm."""
    if algorithm == "dqn":
        return DQNAgent(
            n_actions=n_actions,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            target_update_interval=config["target_update_interval"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exploration_fraction"],
            device=device
        )
    elif algorithm == "ppo":
        return PPOAgent(
            n_actions=n_actions,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            n_epochs=config["n_epochs"],
            n_steps=config["n_steps"],
            n_envs=config["n_envs"],
            batch_size=config["batch_size"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            normalize_advantage=config["normalize_advantage"],
            device=device
        )
    elif algorithm == "a2c":
        return A2CAgent(
            n_actions=n_actions,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            n_steps=config["n_steps"],
            n_envs=config["n_envs"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            normalize_advantage=config["normalize_advantage"],
            use_rms_prop=config["use_rms_prop"],
            rms_prop_eps=config["rms_prop_eps"],
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Ms. Pac-Man")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dqn", "ppo", "a2c"],
        required=True,
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/{algorithm}_config.yaml)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total training steps (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for model checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config is None:
        args.config = f"configs/{args.algorithm}_config.yaml"
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.steps is not None:
        config["total_timesteps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device != "auto":
        config["device"] = args.device
    
    # Set device
    device = get_device(config["device"])
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config["seed"])
    
    # Create environment
    if args.algorithm == "dqn":
        env = create_env(config["env_name"], seed=config["seed"])
        eval_env = create_env(config["env_name"], seed=config["seed"] + 1000)
    else:
        env = create_vec_env(config["env_name"], n_envs=config["n_envs"], seed=config["seed"])
        eval_env = create_env(config["env_name"], seed=config["seed"] + 1000)
    
    # Get action space size
    if hasattr(env, "single_action_space"):
        n_actions = env.single_action_space.n
    else:
        n_actions = env.action_space.n
    
    print(f"Environment: {config['env_name']}")
    print(f"Action space: {n_actions}")
    
    # Create agent
    agent = create_agent(args.algorithm.lower(), config, n_actions, device)
    print(f"Agent: {args.algorithm.upper()}")
    
    # Create logger and checkpoint manager
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.algorithm}_{timestamp}")
    logger = Logger(log_dir, args.algorithm.upper())
    
    checkpoint_manager = CheckpointManager(args.model_dir, args.algorithm.upper())
    
    # Create trainer
    if args.algorithm == "dqn":
        trainer = DQNTrainer(
            agent=agent,
            env=env,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            total_timesteps=config["total_timesteps"],
            learning_starts=config["learning_starts"],
            train_freq=config["train_freq"],
            eval_env=eval_env,
            eval_freq=config.get("eval_freq", 10000),
            eval_episodes=config.get("eval_episodes", 10),
            save_freq=config.get("save_freq", 50000),
            log_interval=config.get("log_interval", 1000)
        )
    else:
        trainer = OnPolicyTrainer(
            agent=agent,
            env=env,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            total_timesteps=config["total_timesteps"],
            eval_env=eval_env,
            eval_freq=config.get("eval_freq", 10000),
            eval_episodes=config.get("eval_episodes", 10),
            save_freq=config.get("save_freq", 50000),
            log_interval=config.get("log_interval", 1)
        )
    
    # Start training
    print(f"\nStarting training for {config['total_timesteps']:,} steps...")
    print(f"Logs: {log_dir}")
    print(f"Models: {args.model_dir}\n")
    
    try:
        trainer.train()
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
