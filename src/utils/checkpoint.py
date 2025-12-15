"""Checkpoint management for saving and loading models."""

import os
import torch
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints.
    """
    
    def __init__(self, save_dir: str, algorithm: str):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            algorithm: Name of the algorithm
        """
        self.save_dir = save_dir
        self.algorithm = algorithm
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        episode: int,
        reward: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            episode: Current episode number
            reward: Current reward (for naming)
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.algorithm}_step{step}_reward{reward:.0f}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "episode": episode,
            "reward": reward,
            "algorithm": self.algorithm,
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary with checkpoint metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
        print(f"  Episode: {checkpoint.get('episode', 'N/A')}")
        print(f"  Reward: {checkpoint.get('reward', 'N/A')}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = [
            f for f in os.listdir(self.save_dir)
            if f.startswith(self.algorithm) and f.endswith(".pt")
        ]
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)))
        
        return os.path.join(self.save_dir, checkpoints[-1])
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        reward: float
    ) -> str:
        """
        Save the best model based on reward.
        
        Args:
            model: Model to save
            reward: Reward achieved
            
        Returns:
            Path to saved model
        """
        filename = f"{self.algorithm}_best_reward{reward:.0f}.pt"
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "reward": reward,
            "algorithm": self.algorithm,
        }, filepath)
        
        print(f"Best model saved: {filepath}")
        
        return filepath
