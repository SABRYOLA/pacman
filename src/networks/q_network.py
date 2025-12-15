"""Deep Q-Network architecture for DQN agent."""

import torch
import torch.nn as nn
from typing import Tuple


class QNetwork(nn.Module):
    """
    Convolutional Neural Network for Deep Q-Learning.
    
    Architecture:
        - 3 convolutional layers for feature extraction
        - 2 fully connected layers for Q-value approximation
        - Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
        - Output: (batch, n_actions) - Q-values for each action
    """
    
    def __init__(self, n_actions: int, input_shape: Tuple[int, int, int] = (4, 84, 84)):
        """
        Initialize Q-Network.
        
        Args:
            n_actions: Number of possible actions
            input_shape: Shape of input (channels, height, width)
        """
        super(QNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size of flattened features
        conv_out_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.conv(dummy_input)
            return int(dummy_output.numel())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Q-values for each action (batch, n_actions)
        """
        # Normalize input to [0, 1]
        x = x.float() / 255.0
        
        # Convolutional features
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        
        # Q-values
        q_values = self.fc(x)
        
        return q_values
