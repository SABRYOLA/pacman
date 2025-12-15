"""Atari environment preprocessing wrappers."""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Any, Dict, Tuple


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset."""
    
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max over last 2 frames and skip frames."""
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over."""
    
    def __init__(self, env: gym.Env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Lost a life
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    
    def __init__(self, env: gym.Env):
        super(FireResetEnv, self).__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert action_meanings[1] == 'FIRE'
        assert len(action_meanings) >= 3
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True):
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        if self.grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
    
    def observation(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class FrameStack(gym.Wrapper):
    """Stack k last frames."""
    
    def __init__(self, env: gym.Env, k: int = 4):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, shp[0], shp[1]), dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        return np.array(self.frames)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, 1}."""
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)


def make_atari_env(
    env_id: str,
    seed: int = 0,
    frame_stack: int = 4,
    clip_rewards: bool = True,
    episodic_life: bool = True
) -> gym.Env:
    """
    Create Atari environment with standard preprocessing wrappers.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/MsPacman-v5")
        seed: Random seed
        frame_stack: Number of frames to stack
        clip_rewards: Whether to clip rewards to {-1, 0, 1}
        episodic_life: Whether to use episodic life wrapper
        
    Returns:
        Wrapped Gymnasium environment
    """
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = WarpFrame(env)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    env = FrameStack(env, k=frame_stack)
    
    # Set seed
    env.action_space.seed(seed)
    
    return env
