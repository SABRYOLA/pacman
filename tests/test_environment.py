"""Tests for environment wrappers."""

import pytest
import numpy as np
import gymnasium as gym

from src.environment import make_atari_env, create_env, create_vec_env


class TestEnvironmentWrappers:
    """Tests for Atari environment wrappers."""
    
    def test_make_atari_env(self):
        """Test Atari environment creation with wrappers."""
        env = make_atari_env("ALE/Pong-v5", seed=42)
        
        # Check observation space
        assert env.observation_space.shape == (4, 84, 84)
        assert env.observation_space.dtype == np.uint8
        
        # Check action space
        assert isinstance(env.action_space, gym.spaces.Discrete)
        
        env.close()
    
    def test_environment_step(self):
        """Test environment step returns correct format."""
        env = make_atari_env("ALE/Pong-v5", seed=42)
        
        obs, info = env.reset()
        assert obs.shape == (4, 84, 84)
        assert isinstance(info, dict)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4, 84, 84)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        
        env.close()
    
    def test_frame_stacking(self):
        """Test frame stacking produces correct shape."""
        env = make_atari_env("ALE/Pong-v5", seed=42, frame_stack=4)
        
        obs, _ = env.reset()
        assert obs.shape == (4, 84, 84)
        
        env.close()
    
    def test_reward_clipping(self):
        """Test reward clipping."""
        env = make_atari_env("ALE/Pong-v5", seed=42, clip_rewards=True)
        
        obs, _ = env.reset()
        
        # Play until we get a non-zero reward
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # If we got a reward, check it's clipped
            if reward != 0:
                assert reward in [-1, 1]
                break
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()


class TestEnvironmentFactory:
    """Tests for environment factory functions."""
    
    def test_create_env(self):
        """Test single environment creation."""
        env = create_env("ALE/Pong-v5", seed=42)
        
        assert env.observation_space.shape == (4, 84, 84)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        
        env.close()
    
    def test_create_vec_env(self):
        """Test vectorized environment creation."""
        n_envs = 4
        vec_env = create_vec_env("ALE/Pong-v5", n_envs=n_envs, seed=42)
        
        # Check it's a vector environment
        assert hasattr(vec_env, "num_envs")
        assert vec_env.num_envs == n_envs
        
        # Test reset
        obs, info = vec_env.reset()
        assert obs.shape == (n_envs, 4, 84, 84)
        
        # Test step
        actions = np.array([vec_env.single_action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminates, truncates, infos = vec_env.step(actions)
        
        assert obs.shape == (n_envs, 4, 84, 84)
        assert rewards.shape == (n_envs,)
        assert terminates.shape == (n_envs,)
        assert truncates.shape == (n_envs,)
        
        vec_env.close()


if __name__ == "__main__":
    pytest.main([__file__])
