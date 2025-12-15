"""Main training loop and utilities."""

import numpy as np
import torch
from typing import Optional, Callable
from tqdm import tqdm
import gymnasium as gym


class Trainer:
    """
    Generic trainer for reinforcement learning agents.
    """
    
    def __init__(
        self,
        agent,
        env: gym.Env,
        logger,
        checkpoint_manager,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        eval_episodes: int = 10,
        save_freq: int = 50000,
        log_interval: int = 1000,
    ):
        """
        Initialize trainer.
        
        Args:
            agent: RL agent (DQN, PPO, or A2C)
            env: Training environment
            logger: Logger for metrics
            checkpoint_manager: Checkpoint manager
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            eval_freq: Frequency of evaluation
            eval_episodes: Number of episodes per evaluation
            save_freq: Frequency of saving checkpoints
            log_interval: Frequency of logging
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.total_timesteps = total_timesteps
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_freq = save_freq
        self.log_interval = log_interval
        
        self.best_mean_reward = -np.inf
    
    def train(self) -> None:
        """Run training loop."""
        raise NotImplementedError("Subclass must implement train()")
    
    def evaluate(self) -> float:
        """
        Evaluate the agent.
        
        Returns:
            Mean episode reward
        """
        if self.eval_env is None:
            return 0.0
        
        episode_rewards = []
        
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(obs, training=False)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        
        # Log evaluation results
        self.logger.log_scalar("eval/mean_reward", mean_reward)
        self.logger.log_scalar("eval/std_reward", np.std(episode_rewards))
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.checkpoint_manager.save_best_model(
                self.agent.q_network if hasattr(self.agent, 'q_network') else self.agent.policy,
                mean_reward
            )
        
        return mean_reward


class DQNTrainer(Trainer):
    """Trainer specifically for DQN agent."""
    
    def __init__(
        self,
        agent,
        env: gym.Env,
        logger,
        checkpoint_manager,
        total_timesteps: int,
        learning_starts: int = 50000,
        train_freq: int = 4,
        **kwargs
    ):
        super().__init__(agent, env, logger, checkpoint_manager, total_timesteps, **kwargs)
        self.learning_starts = learning_starts
        self.train_freq = train_freq
    
    def train(self) -> None:
        """Run DQN training loop."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        with tqdm(total=self.total_timesteps, desc="DQN Training") as pbar:
            for step in range(self.total_timesteps):
                # Update epsilon
                progress = step / self.total_timesteps
                self.agent.update_epsilon(progress)
                
                # Select action
                action = self.agent.select_action(obs, training=True)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(obs, action, reward, next_obs, done)
                
                episode_reward += reward
                episode_length += 1
                
                # Train
                if step >= self.learning_starts and step % self.train_freq == 0:
                    loss = self.agent.train_step()
                    if loss is not None and step % self.log_interval == 0:
                        self.logger.log_scalar("train/loss", loss)
                        self.logger.log_scalar("train/epsilon", self.agent.epsilon)
                
                # Episode end
                if done:
                    self.logger.log_episode(episode_reward, episode_length)
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs
                
                # Update step counter
                self.logger.total_steps = step + 1
                
                # Logging
                if (step + 1) % self.log_interval == 0:
                    self.logger.print_stats()
                
                # Evaluation
                if self.eval_env is not None and (step + 1) % self.eval_freq == 0:
                    mean_reward = self.evaluate()
                    print(f"Eval mean reward: {mean_reward:.2f}")
                
                # Save checkpoint
                if (step + 1) % self.save_freq == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.agent.q_network,
                        self.agent.optimizer,
                        step + 1,
                        self.logger.num_episodes,
                        self.logger.get_stats().get("mean_reward", 0)
                    )
                
                pbar.update(1)
        
        self.logger.close()


class OnPolicyTrainer(Trainer):
    """Trainer for on-policy algorithms (PPO, A2C)."""
    
    def train(self) -> None:
        """Run on-policy training loop."""
        # Vectorized environment setup
        obs, _ = self.env.reset()
        n_envs = obs.shape[0] if len(obs.shape) > 3 else 1
        
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs)
        
        num_updates = self.total_timesteps // (self.agent.n_steps * self.agent.n_envs)
        
        with tqdm(total=self.total_timesteps, desc=f"{self.logger.algorithm} Training") as pbar:
            for update in range(num_updates):
                # Collect rollout
                for step in range(self.agent.n_steps):
                    # Select actions
                    actions, values, log_probs = self.agent.select_action(obs)
                    
                    # Step environments
                    next_obs, rewards, terminates, truncateds, infos = self.env.step(actions)
                    dones = np.logical_or(terminates, truncateds)
                    
                    # Store transition
                    self.agent.store_transition(obs, actions, rewards, dones, values, log_probs)
                    
                    # Update episode stats
                    episode_rewards += rewards
                    episode_lengths += 1
                    
                    # Log completed episodes
                    for i, done in enumerate(dones):
                        if done:
                            self.logger.log_episode(episode_rewards[i], episode_lengths[i])
                            episode_rewards[i] = 0
                            episode_lengths[i] = 0
                    
                    obs = next_obs
                    
                    # Update step counter
                    self.logger.total_steps += n_envs
                    pbar.update(n_envs)
                
                # Train
                metrics = self.agent.train_step(obs)
                self.logger.log_training_metrics(metrics)
                
                # Logging
                if (update + 1) % self.log_interval == 0:
                    self.logger.print_stats()
                
                # Evaluation
                if self.eval_env is not None and (update + 1) % (self.eval_freq // (self.agent.n_steps * self.agent.n_envs)) == 0:
                    mean_reward = self.evaluate()
                    print(f"Eval mean reward: {mean_reward:.2f}")
                
                # Save checkpoint
                if (update + 1) % (self.save_freq // (self.agent.n_steps * self.agent.n_envs)) == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.agent.policy,
                        self.agent.optimizer,
                        self.logger.total_steps,
                        self.logger.num_episodes,
                        self.logger.get_stats().get("mean_reward", 0)
                    )
        
        self.logger.close()
