"""Factory functions for creating environments."""

import gymnasium as gym
from typing import Optional, Callable
from .wrappers import make_atari_env


def create_env(
    env_name: str,
    seed: int = 0,
    render_mode: Optional[str] = None,
    **kwargs
) -> gym.Env:
    """
    Create an environment with appropriate wrappers.
    
    Args:
        env_name: Name of the environment
        seed: Random seed
        render_mode: Rendering mode (None, "human", "rgb_array")
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Wrapped environment
    """
    if "Pacman" in env_name or "ALE" in env_name:
        # Atari environment
        return make_atari_env(env_name, seed=seed, **kwargs)
    else:
        # Generic environment
        env = gym.make(env_name, render_mode=render_mode)
        env.action_space.seed(seed)
        return env


def create_vec_env(
    env_name: str,
    n_envs: int = 4,
    seed: int = 0,
    **kwargs
) -> gym.vector.VectorEnv:
    """
    Create vectorized environments for parallel training.
    
    Args:
        env_name: Name of the environment
        n_envs: Number of parallel environments
        seed: Base random seed
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Vectorized environment
    """
    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            return create_env(env_name, seed=seed + rank, **kwargs)
        return _init
    
    env_fns = [make_env(i) for i in range(n_envs)]
    return gym.vector.AsyncVectorEnv(env_fns)
