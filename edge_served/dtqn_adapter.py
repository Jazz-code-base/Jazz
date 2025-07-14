import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import os
import json
import datetime
from enum import Enum

from dtqn.agents.dtqn import DtqnAgent
from dtqn.networks.dtqn import DTQN_adapter as DTQN_Network
from dtqn.networks.dual_flow_dtqn import DualFlowDTQN
from utils.random import RNG
from dtqn.buffers.replay_buffer import ReplayBuffer
from dtqn.buffers.per_buffer import PERBuffer
from utils.bag import Bag

# Define training mode enum, consistent with TrainMode in DTQN
class TrainMode(Enum):
    TRAIN = 0
    EVAL = 1

# Create an adapter class that maintains the same interface as the original DDQN
class DTQN_adapter:
    def __init__(self, cfg):
        self.n_actions = cfg['n_actions']
        self.device = torch.device(cfg['device'])
        self.gamma = cfg['gamma']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.lr = cfg['lr']
        self.lr_decay = cfg['lr_decay']
        self.lr_decay_train_counts = cfg['lr_decay_train_counts']
        self.lr_min = cfg['lr_min']
        self.target_update_count = cfg['target_update_count']
        self.sample_count = 0
        self.train_count = 0
        self.epsilon = self.epsilon_start
        self.model_save_interval = cfg['model_save_interval']
        
        # Use fixed model save directory
        self.model_save_dir = "saved_models"
        os.makedirs(self.model_save_dir, exist_ok=True)
        print(f"Models will be saved to fixed directory: {self.model_save_dir}")
        
        # DTQN specific configuration
        self.state_dim = cfg['n_states']
        self.history_len = cfg.get('history_len', 10)
        self.context_len = cfg.get('context_len', 10)
        self.bag_size = cfg.get('bag_size', 5)
        self.obs_mask = 0.0
        self.max_env_steps = cfg.get('max_env_steps', 300)
        self.hidden_dim = cfg.get('hidden_dim', 64) # Note: This might still be unused by the networks themselves
        
        # Priority experience replay configuration
        self.use_per = cfg.get('use_per', False)
        self.per_alpha = cfg.get('per_alpha', 0.6)
        self.per_beta_start = cfg.get('per_beta_start', 0.4)
        self.per_beta_frames = cfg.get('per_beta_frames', 100000)
        self.per_eps = cfg.get('per_eps', 1e-6)
        
        # Network configuration
        self.use_dual_flow = cfg.get('use_dual_flow', True) # Controlled by server.py
        
        # Create DTQN network
        self.dtqn_network = self._create_dtqn_network(cfg)
        
        # Create DTQN agent
        self.dtqn_agent = self._create_dtqn_agent(cfg)
        
        # Set optimizer
        self.optimizer = optim.Adam(self.dtqn_agent.policy_network.parameters(), lr=self.lr)
        self.dtqn_agent.optimizer = self.optimizer
        
        # Provide reference to internal buffer for compatibility with original interface
        self.memory = self.dtqn_agent.replay_buffer
        
        # For storing previous state
        self.prev_state = None
        
        # Ensure RNG is initialized
        if not hasattr(RNG, 'rng') or RNG.rng is None:
            RNG.rng = np.random.default_rng()
        
        # Final check: ensure all network components are on the correct device
        self._ensure_device_placement()
        
    def _ensure_device_placement(self):
        """Ensure all network components are on the correct device"""
        print(f"Checking network device placement, target device: {self.device}")
        
        # Check and move policy network
        if hasattr(self.dtqn_agent, 'policy_network') and self.dtqn_agent.policy_network is not None:
            self.dtqn_agent.policy_network = self.dtqn_agent.policy_network.to(self.device)
            # Check if the first parameter is on the correct device
            first_param = next(self.dtqn_agent.policy_network.parameters(), None)
            if first_param is not None:
                print(f"Policy network device: {first_param.device}")
            else:
                print("Policy network has no parameters")
        
        # Check and move target network
        if hasattr(self.dtqn_agent, 'target_network') and self.dtqn_agent.target_network is not None:
            self.dtqn_agent.target_network = self.dtqn_agent.target_network.to(self.device)
            # Check if the first parameter is on the correct device
            first_param = next(self.dtqn_agent.target_network.parameters(), None)
            if first_param is not None:
                print(f"Target network device: {first_param.device}")
            else:
                print("Target network has no parameters")
        
        # Check and move dtqn_network (if needed)
        if hasattr(self, 'dtqn_network') and self.dtqn_network is not None:
            self.dtqn_network = self.dtqn_network.to(self.device)
            first_param = next(self.dtqn_network.parameters(), None)
            if first_param is not None:
                print(f"DTQN network device: {first_param.device}")
        
        print("Network device placement check completed")
        
    def _create_dtqn_network(self, cfg):
        """
        Create DTQN network, choose specific implementation based on use_dual_flow parameter.
        """
        network_cfg = {
            'obs_dim': self.state_dim,
            'num_actions': self.n_actions,
            'embed_per_obs_dim': cfg.get('embed_per_obs_dim', 32),
            'action_embed_dim': cfg.get('action_embed_dim', 16),
            'inner_embed_size': cfg.get('inner_embed_size', 128),
            'num_heads': cfg.get('num_heads', 2),
            'num_transformer_layers': cfg.get('num_transformer_layers', 2),
            'dropout': cfg.get('dropout', 0.1),
            'gate': cfg.get('gate', "res"),
            'identity': cfg.get('identity', False),
            'pos': cfg.get('pos', "sin"), # DTQN_Network uses this, DualFlowDTQN defaults to "learned" if not overridden
            'history_len': self.history_len,
            'discrete': cfg.get('discrete', False),
            'bag_size': self.bag_size,
            # Parameter specific to DualFlowDTQN, DTQN_Network will ignore it if passed via **kwargs
            'subflow_hidden_dim': cfg.get('subflow_hidden_dim', 64) 
        }
            
        if self.use_dual_flow:
            # print("Using DualFlowDTQN network")
            return DualFlowDTQN(**network_cfg)
        else:
            print("Using DTQN_Network (dtqn.networks.dtqn.DTQN_adapter)")
            # DTQN_Network might not use subflow_hidden_dim, but it's harmless to pass it if it handles **kwargs
            return DTQN_Network(**network_cfg)
    
    def _create_dtqn_agent(self, cfg):
        """Create DTQN agent"""
        # Network factory function
        def network_factory():
            network = self._create_dtqn_network(cfg)
            # Ensure network is moved to correct device
            network = network.to(self.device)
            return network
        
        # Choose buffer type
        if self.use_per:
            # Create priority experience replay buffer
            replay_buffer = PERBuffer(
                buffer_size=cfg.get('memory_capacity', 100000),
                env_obs_length=self.state_dim,
                obs_mask=self.obs_mask,
                max_episode_steps=self.max_env_steps,
                context_len=self.context_len,
                alpha=self.per_alpha,
                beta_start=self.per_beta_start,
                beta_frames=self.per_beta_frames,
                eps=self.per_eps
            )
            print(f"Using priority experience replay buffer, parameters: alpha={self.per_alpha}, beta_start={self.per_beta_start}")
        else:
            # Create standard replay buffer
            replay_buffer = ReplayBuffer(
                buffer_size=cfg.get('memory_capacity', 100000),
                env_obs_length=self.state_dim,
                obs_mask=self.obs_mask,
                max_episode_steps=self.max_env_steps,
                context_len=self.context_len
            )
            print("Using standard replay buffer")
        
        # Create agent
        agent = DtqnAgent(
            network_factory=network_factory,
            buffer_size=cfg.get('memory_capacity', 100000),
            device=self.device,
            env_obs_length=self.state_dim,
            max_env_steps=self.max_env_steps,
            obs_mask=self.obs_mask,
            num_actions=self.n_actions,
            is_discrete_env=False,
            learning_rate=self.lr,
            batch_size=self.batch_size,
            context_len=self.context_len,
            gamma=self.gamma,
            grad_norm_clip=1.0,
            target_update_frequency=self.target_update_count,
            history=self.history_len,
            bag_size=self.bag_size
        )
        
        # Replace default buffer with our created buffer
        agent.replay_buffer = replay_buffer
        
        # Ensure networks are moved to correct device
        if hasattr(agent, 'policy_network') and agent.policy_network is not None:
            agent.policy_network = agent.policy_network.to(self.device)
            print(f"Policy network moved to device: {self.device}")
        
        if hasattr(agent, 'target_network') and agent.target_network is not None:
            agent.target_network = agent.target_network.to(self.device)
            print(f"Target network moved to device: {self.device}")
        
        return agent
    
    def sample_action(self, state):
        """Sample action
        
        Returns:
            tuple: (action, whether it's a random action)
        """
        # Ensure input state is on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        
        # Only increase sample_count when agent can sample from buffer for training (used for epsilon decay)
        if self.dtqn_agent.replay_buffer.can_sample(self.batch_size):
            self.sample_count += 1
        
        # Recalculate epsilon based on possibly updated sample_count
        # If sample_count hasn't changed, epsilon maintains its previous value
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        
        # If this is the first state, reset context
        if self.prev_state is None:
            # Ensure using .cpu() when converting to numpy
            state_np = state.detach().cpu().numpy() if state.requires_grad else state.cpu().numpy()
            self.dtqn_agent.context_reset(state_np)
        
        # Get action using DTQN agent
        action, is_random = self.dtqn_agent.get_action(epsilon=self.epsilon)
        
        # Ensure action is within valid range
        action = min(max(action, 0), self.n_actions - 1)
        
        # Save current state
        self.prev_state = state
        
        return action, is_random
    
    def predict_action(self, state):
        """Predict action without exploration
        
        Returns:
            tuple: (action, whether it's a random action)
        """
        # Ensure input state is on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
            
        if self.prev_state is None:
            # Ensure using .cpu() when converting to numpy
            state_np = state.detach().cpu().numpy() if state.requires_grad else state.cpu().numpy()
            self.dtqn_agent.context_reset(state_np)
        
        action, is_random = self.dtqn_agent.get_action(epsilon=0.0)
        self.prev_state = state
        
        return action, is_random
    
    def update(self):
        """Update model, maintain same interface as original DDQN"""
        # Simplify logging to avoid duplication with server.py
        
        # First check if internal buffer has enough samples
        can_sample = False
        if hasattr(self.dtqn_agent, 'replay_buffer') and hasattr(self.dtqn_agent.replay_buffer, 'can_sample'):
            can_sample = self.dtqn_agent.replay_buffer.can_sample(self.batch_size)
        
        # If not enough samples, return False directly
        if not can_sample:
            return False
        
        # Call DTQN agent's training method
        train_success = False
        
        # Actually perform training
        self.dtqn_agent.train()
        
        # Only increase train_count and decay learning rate after successful training
        self.train_count += 1
        train_success = True
        
        # Check if learning rate decay is needed after successful training
        if self.train_count % self.lr_decay_train_counts == 0:
            self._decay_learning_rate()
                
        
        # Periodically save model (only when training is successful)
        if self.train_count % self.model_save_interval == 0:
            self.save_model()
        
        return train_success
    
    def observe(self, state, action, reward, next_state, done):
        """Observe transition and add it to DTQN agent's context"""
        # Ensure all input tensors are on correct device and convert to numpy arrays
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy() if state.requires_grad else state.cpu().numpy()
        else:
            state_np = np.array(state, dtype=np.float32)
            
        if isinstance(next_state, torch.Tensor):
            next_state_np = next_state.detach().cpu().numpy() if next_state.requires_grad else next_state.cpu().numpy()
        else:
            next_state_np = np.array(next_state, dtype=np.float32)
        
        # Process action - ensure it's a scalar
        if isinstance(action, torch.Tensor):
            action = action.item() if action.numel() == 1 else int(action.cpu().numpy())
        else:
            action = int(action)
        
        # Process reward
        if isinstance(reward, torch.Tensor):
            reward = reward.item() if reward.numel() == 1 else float(reward.cpu().numpy())
        else:
            reward = float(reward)
        
        # Process done
        if isinstance(done, torch.Tensor):
            done = done.item() if done.numel() == 1 else bool(done.cpu().numpy())
        else:
            done = bool(done)
        
        # Ensure action index is within valid range
        action = min(max(action, 0), self.n_actions - 1)
        
        # If this is the first state, reset context
        if self.prev_state is None:
            self.dtqn_agent.context_reset(state_np)
        
        # Add transition to DTQN agent's context
        self.dtqn_agent.observe(next_state_np, action, reward, done)
        
        # Update prev_state - ensure it's on correct device
        if isinstance(next_state, torch.Tensor):
            self.prev_state = next_state.to(self.device)
        else:
            self.prev_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
    
    def save_model(self, path=None):
        """Save model to fixed directory, only keep recent versions"""
        # If specific path is provided, use it
        if path is not None:
            save_path = path
        else:
            # Generate filename and path
            filename = f"dtqn_model_{self.train_count}.pth"
            save_path = os.path.join(self.model_save_dir, filename)
        
        # Create save directory (theoretically created in __init__, this is double insurance)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Save model parameters
        save_dict = {
            'policy_network': self.dtqn_agent.policy_network.state_dict(),
            'target_network': self.dtqn_agent.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_count': self.train_count,
            'epsilon': self.epsilon,
            'sample_count': self.sample_count,
        }
        
        # Save priority experience replay parameters (if used)
        if self.use_per and hasattr(self.dtqn_agent.replay_buffer, 'frame_idx'):
            save_dict['per_frame_idx'] = self.dtqn_agent.replay_buffer.frame_idx
            save_dict['per_max_priority'] = self.dtqn_agent.replay_buffer.max_priority
        
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}, training count: {self.train_count}")
        
        # Save a copy of the latest version
        latest_path = os.path.join(self.model_save_dir, "dtqn_model_latest.pth")
        torch.save(save_dict, latest_path)
        print(f"Latest model copy saved to {latest_path}")
        
        # Also save configuration information
        config = {
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'target_update_count': self.target_update_count,
            'history_len': self.history_len,
            'context_len': self.context_len,
            'bag_size': self.bag_size,
            'state_dim': self.state_dim,
            'use_per': self.use_per,
            'train_count': self.train_count,
        }
        
        # Save configuration to same directory
        latest_config_path = os.path.join(self.model_save_dir, "dtqn_model_latest_config.json")
        with open(latest_config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_model(self, path):
        """Load model"""
        if not os.path.exists(path):
            print(f"Warning: Model file {path} does not exist!")
            return False
        
        try:
            # Load model parameters
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load network parameters
            self.dtqn_agent.policy_network.load_state_dict(checkpoint['policy_network'])
            self.dtqn_agent.target_network.load_state_dict(checkpoint['target_network'])
            
            # Load optimizer parameters
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load training state
            self.train_count = checkpoint.get('train_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.sample_count = checkpoint.get('sample_count', 0)
            
            # Load priority experience replay parameters (if used)
            if self.use_per and hasattr(self.dtqn_agent.replay_buffer, 'frame_idx'):
                if 'per_frame_idx' in checkpoint:
                    self.dtqn_agent.replay_buffer.frame_idx = checkpoint['per_frame_idx']
                if 'per_max_priority' in checkpoint:
                    self.dtqn_agent.replay_buffer.max_priority = checkpoint['per_max_priority']
            
            print(f"Successfully loaded model: {path}, training count: {self.train_count}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _decay_learning_rate(self):
        """Decay learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.lr_decay, self.lr_min)
            print(f"Learning rate decayed to: {param_group['lr']:.6f}")

# # Keep MLP class for compatibility with existing test code, but it won't be used
# class MLP(nn.Module):
#     """
#     Placeholder class for interface compatibility
#     Not actually used in DTQN architecture
#     """
#     def __init__(self, n_states, n_actions, hidden_dim=128):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(n_states, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, n_actions)
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # Keep ReplayBuffer class for compatibility with existing test code, but actually use DTQN's internal buffer
# class ReplayBuffer:
#     """
#     Placeholder class for interface compatibility
#     In DTQN architecture, DTQN's internal buffer is actually used
#     """
#     def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.beta_start = beta_start
#         self.beta_frames = beta_frames
#         self.epsilon = 1e-5
#         self.data = []
#         self.pos = 0
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.frame = 1
#         self.n_entries = 0
    
#     def push(self, transition, priority=None):
#         """Add transition to buffer"""
#         print("Warning: Using compatibility ReplayBuffer.push(), but it won't take effect in DTQN")
#         if len(self.data) < self.capacity:
#             self.data.append(transition)
#         else:
#             self.data[self.pos] = transition
        
#         if priority is None:
#             priority = 1.0 if self.n_entries == 0 else self.priorities.max()
        
#         self.priorities[self.pos] = priority ** self.alpha
#         self.pos = (self.pos + 1) % self.capacity
#         self.n_entries = min(self.n_entries + 1, self.capacity)
#         self.frame += 1
    
#     def sample(self, batch_size):
#         """Sample from buffer"""
#         print("Warning: Using compatibility ReplayBuffer.sample(), but it won't take effect in DTQN")
#         if self.n_entries == 0:
#             return [], [], [], [], [], []
        
#         indices = np.random.choice(self.n_entries, batch_size, replace=False)
#         batch = [self.data[idx] for idx in indices]
        
#         state, action, reward, next_state, done = zip(*batch)
#         return indices, np.ones(batch_size), state, action, reward, next_state, done
    
#     def update_priorities(self, indices, priorities):
#         """Update priorities"""
#         print("Warning: Using compatibility ReplayBuffer.update_priorities(), but it won't take effect in DTQN")
#         for idx, priority in zip(indices, priorities):
#             self.priorities[idx] = (priority + self.epsilon) ** self.alpha
    
#     def __len__(self):
#         return self.n_entries 