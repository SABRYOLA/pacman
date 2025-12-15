"""Tests for neural network architectures."""

import pytest
import torch
import numpy as np

from src.networks import QNetwork, ActorCriticNetwork, init_weights, get_device


class TestQNetwork:
    """Tests for Q-Network."""
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        n_actions = 9
        batch_size = 4
        
        network = QNetwork(n_actions)
        input_tensor = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
        
        output = network(input_tensor)
        
        assert output.shape == (batch_size, n_actions)
        assert not torch.isnan(output).any()
    
    def test_different_input_shapes(self):
        """Test network handles different valid input shapes."""
        network = QNetwork(9, input_shape=(4, 84, 84))
        
        # Single sample
        single = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8)
        output = network(single)
        assert output.shape == (1, 9)
        
        # Batch
        batch = torch.randint(0, 256, (16, 4, 84, 84), dtype=torch.uint8)
        output = network(batch)
        assert output.shape == (16, 9)


class TestActorCriticNetwork:
    """Tests for Actor-Critic Network."""
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        n_actions = 9
        batch_size = 4
        
        network = ActorCriticNetwork(n_actions)
        input_tensor = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
        
        action_logits, values = network(input_tensor)
        
        assert action_logits.shape == (batch_size, n_actions)
        assert values.shape == (batch_size, 1)
        assert not torch.isnan(action_logits).any()
        assert not torch.isnan(values).any()
    
    def test_get_action_and_value(self):
        """Test action sampling and value estimation."""
        n_actions = 9
        batch_size = 4
        
        network = ActorCriticNetwork(n_actions)
        input_tensor = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
        
        action, log_prob, entropy, value = network.get_action_and_value(input_tensor)
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        
        # Check action bounds
        assert (action >= 0).all() and (action < n_actions).all()
    
    def test_get_value_only(self):
        """Test value-only estimation."""
        network = ActorCriticNetwork(9)
        input_tensor = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
        
        value = network.get_value(input_tensor)
        
        assert value.shape == (4, 1)


class TestUtilities:
    """Tests for network utilities."""
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # Should be cuda if available, else cpu
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"
    
    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_init_weights(self):
        """Test weight initialization."""
        layer = torch.nn.Linear(10, 10)
        
        # Initialize weights
        init_weights(layer, gain=1.0)
        
        # Check weights are not zero
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        
        # Check bias is zero
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))


if __name__ == "__main__":
    pytest.main([__file__])
