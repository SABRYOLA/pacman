"""Actor-Critic network architecture for PPO and A2C agents."""

import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Shared CNN backbone with separate actor (policy) and critic (value) heads.
    
    Architecture:
        - Shared convolutional layers for feature extraction
        - Actor head: outputs action probabilities
        - Critic head: outputs state value estimate
        - Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
        - Output: action distribution and value estimate
    """
    
    def __init__(self, n_actions: int, input_shape: Tuple[int, int, int] = (4, 84, 84)):
        """
        Initialize Actor-Critic Network.
        
        Args:
            n_actions: Number of possible actions
            input_shape: Shape of input (channels, height, width)
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.input_shape = input_shape
        
        # Shared convolutional backbone
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
        
        # Shared fully connected layer
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(512, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(512, 1)
    
    def _get_conv_output_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.conv(dummy_input)
            return int(dummy_output.numel())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Tuple of (action_logits, state_values)
        """
        # Normalize input to [0, 1]
        x = x.float() / 255.0
        
        # Shared features
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        
        # Actor output (action logits)
        action_logits = self.actor(x)
        
        # Critic output (state value)
        state_values = self.critic(x)
        
        return action_logits, state_values
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, value, log probability, and entropy.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            action: Optional action tensor for computing log_prob
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(x)
        
        # Create categorical distribution
        probs = Categorical(logits=action_logits)
        
        # Sample action if not provided
        if action is None:
            action = probs.sample()
        
        # Compute log probability and entropy
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the value estimate.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            State value estimate
        """
        _, value = self.forward(x)
        return value
