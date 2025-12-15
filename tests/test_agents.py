"""Tests for RL agents."""

import pytest
import torch
import numpy as np

from src.agents import DQNAgent, PPOAgent, A2CAgent


class TestDQNAgent:
    """Tests for DQN agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(n_actions=9, device=torch.device("cpu"))
        
        assert agent.n_actions == 9
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.optimizer is not None
        assert agent.replay_buffer is not None
    
    def test_select_action(self):
        """Test action selection."""
        agent = DQNAgent(n_actions=9, device=torch.device("cpu"))
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        
        action = agent.select_action(obs, training=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 9
    
    def test_store_transition(self):
        """Test storing transitions."""
        agent = DQNAgent(n_actions=9, device=torch.device("cpu"))
        
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        action = 0
        reward = 1.0
        next_obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        done = False
        
        agent.store_transition(obs, action, reward, next_obs, done)
        
        assert len(agent.replay_buffer) == 1
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        agent = DQNAgent(
            n_actions=9,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            exploration_fraction=0.1,
            device=torch.device("cpu")
        )
        
        # At start
        agent.update_epsilon(0.0)
        assert agent.epsilon == 1.0
        
        # Midway through decay
        agent.update_epsilon(0.05)
        assert 0.01 < agent.epsilon < 1.0
        
        # After decay
        agent.update_epsilon(0.5)
        assert agent.epsilon == 0.01


class TestPPOAgent:
    """Tests for PPO agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(n_actions=9, n_envs=4, device=torch.device("cpu"))
        
        assert agent.n_actions == 9
        assert agent.n_envs == 4
        assert agent.policy is not None
        assert agent.optimizer is not None
        assert agent.rollout_buffer is not None
    
    def test_select_action(self):
        """Test action selection for multiple environments."""
        n_envs = 4
        agent = PPOAgent(n_actions=9, n_envs=n_envs, device=torch.device("cpu"))
        
        obs = np.random.randint(0, 256, (n_envs, 4, 84, 84), dtype=np.uint8)
        actions, values, log_probs = agent.select_action(obs, training=True)
        
        assert actions.shape == (n_envs,)
        assert values.shape == (n_envs,)
        assert log_probs.shape == (n_envs,)
        assert (actions >= 0).all() and (actions < 9).all()
    
    def test_store_transition(self):
        """Test storing transitions."""
        n_envs = 4
        agent = PPOAgent(n_actions=9, n_envs=n_envs, n_steps=5, device=torch.device("cpu"))
        
        obs = np.random.randint(0, 256, (n_envs, 4, 84, 84), dtype=np.uint8)
        actions = np.random.randint(0, 9, n_envs)
        rewards = np.random.randn(n_envs)
        dones = np.zeros(n_envs)
        values = np.random.randn(n_envs)
        log_probs = np.random.randn(n_envs)
        
        agent.store_transition(obs, actions, rewards, dones, values, log_probs)
        
        assert agent.rollout_buffer.ptr == 1


class TestA2CAgent:
    """Tests for A2C agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = A2CAgent(n_actions=9, n_envs=4, device=torch.device("cpu"))
        
        assert agent.n_actions == 9
        assert agent.n_envs == 4
        assert agent.policy is not None
        assert agent.optimizer is not None
        assert agent.rollout_buffer is not None
    
    def test_select_action(self):
        """Test action selection for multiple environments."""
        n_envs = 4
        agent = A2CAgent(n_actions=9, n_envs=n_envs, device=torch.device("cpu"))
        
        obs = np.random.randint(0, 256, (n_envs, 4, 84, 84), dtype=np.uint8)
        actions, values, log_probs = agent.select_action(obs, training=True)
        
        assert actions.shape == (n_envs,)
        assert values.shape == (n_envs,)
        assert log_probs.shape == (n_envs,)
        assert (actions >= 0).all() and (actions < 9).all()
    
    def test_store_transition(self):
        """Test storing transitions."""
        n_envs = 4
        agent = A2CAgent(n_actions=9, n_envs=n_envs, n_steps=5, device=torch.device("cpu"))
        
        obs = np.random.randint(0, 256, (n_envs, 4, 84, 84), dtype=np.uint8)
        actions = np.random.randint(0, 9, n_envs)
        rewards = np.random.randn(n_envs)
        dones = np.zeros(n_envs)
        values = np.random.randn(n_envs)
        log_probs = np.random.randn(n_envs)
        
        agent.store_transition(obs, actions, rewards, dones, values, log_probs)
        
        assert agent.rollout_buffer.ptr == 1


if __name__ == "__main__":
    pytest.main([__file__])
