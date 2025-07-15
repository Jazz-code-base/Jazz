from typing import Callable, Union, Tuple
from enum import Enum
import random

import torch
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import joblib

from dtqn.buffers.replay_buffer import ReplayBuffer
from utils.logging_utils import RunningAverage
from utils.context import Context
from utils.epsilon_anneal import LinearAnneal
from utils.random import RNG
from utils.bag import Bag


class TrainMode(Enum):
    TRAIN = 1
    EVAL = 2


class DtqnAgent:
    def __init__(
        self,
        network_factory: Callable[[], Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        is_discrete_env: bool,
        learning_rate: float = 0.0003,
        batch_size: int = 32,
        context_len: int = 50,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        history: int = 50,
        bag_size: int = 0,
        **kwargs,
    ):
        # Basic DQN parameters
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency
        self.num_actions = num_actions
        self.train_mode = TrainMode.TRAIN
        self.obs_mask = -1  # Use -1 as mask value
        
        # DTQN specific parameters
        self.history = history
        self.bag_size = bag_size

        # Initialize networks
        self.policy_network = network_factory()
        self.target_network = network_factory()
        self.target_update()  # Ensure network parameters are identical
        self.target_network.eval()

        # Set observation data types
        if is_discrete_env:
            self.obs_context_type = np.int_
            self.obs_tensor_type = torch.long
        else:
            self.obs_context_type = np.float32
            self.obs_tensor_type = torch.float32

        # PyTorch configuration
        self.device = device
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            obs_mask=self.obs_mask,
            max_episode_steps=max_env_steps,
            context_len=context_len,
        )

        # Initialize context and Bag
        self.train_context = Context(
            context_len,
            self.obs_mask,
            num_actions,
            env_obs_length,
        )
        self.eval_context = Context(
            context_len,
            self.obs_mask,
            num_actions,
            env_obs_length,
        )
        self.train_bag = Bag(bag_size, self.obs_mask, env_obs_length)
        self.eval_bag = Bag(bag_size, self.obs_mask, env_obs_length)

        # Logging
        self.num_train_steps = 0
        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    @property
    def context(self) -> Context:
        """Get context for current mode
        
        Returns:
            Context: Context object for training or evaluation mode
        """
        if self.train_mode == TrainMode.TRAIN:
            return self.train_context
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_context

    @property
    def bag(self) -> Bag:
        """Get Bag for current mode
        
        Returns:
            Bag: Bag object for training or evaluation mode
        """
        if self.train_mode == TrainMode.TRAIN:
            return self.train_bag
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_bag

    def eval_on(self) -> None:
        """Switch to evaluation mode"""
        self.train_mode = TrainMode.EVAL
        self.policy_network.eval()

    def eval_off(self) -> None:
        """Switch to training mode"""
        self.train_mode = TrainMode.TRAIN
        self.policy_network.train()

    @torch.no_grad()
    def get_action(self, epsilon: float = 0.0) -> tuple:
        """Get an ε-greedy action using the policy network
        
        Args:
            epsilon: Exploration probability
            
        Returns:
            tuple: (selected action, whether it's a random action)
        """
        if RNG.rng.random() < epsilon:
            return RNG.rng.integers(self.num_actions), True
        # Truncate context observations and actions to remove padding (if any)
        context_obs_tensor = torch.as_tensor(
            self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
            dtype=self.obs_tensor_type,
            device=self.device,
        ).unsqueeze(0)
        
        # # --- Print state sequence used for inference ---
        # print(f"--- [Inference get_action] State sequence used (context_obs_tensor) ---")
        # print(f"Shape: {context_obs_tensor.shape}")
        # # Print each state in the sequence, note tensor may be on GPU, need to move to CPU and convert to numpy
        # for i in range(context_obs_tensor.shape[1]): # context_obs_tensor is [1, seq_len, obs_dim]
        #     print(f"State {i}: {context_obs_tensor[0, i].cpu().numpy()}")
        # print(f"--- [Inference get_action] End of state sequence print ---")
        # # --- End print ---
        
        # Get context actions and ensure they're within valid range
        context_actions = self.context.action[: min(self.context.max_length, self.context.timestep + 1)].copy()
        context_actions = np.clip(context_actions, 0, self.num_actions - 1)
        
        context_action_tensor = torch.as_tensor(
            context_actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        
        # Always include the full bag, even with padding
        bag_obs_tensor = torch.as_tensor(
            self.bag.obss,
            dtype=self.obs_tensor_type,
            device=self.device
        ).unsqueeze(0)
        
        # Ensure bag actions are within valid range
        bag_actions = np.clip(self.bag.actions, 0, self.num_actions - 1)
        
        bag_action_tensor = torch.as_tensor(
            bag_actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        q_values = self.policy_network(
            context_obs_tensor,
            context_action_tensor,
            bag_obs_tensor,
            bag_action_tensor
        )

        # We take the maximum Q-value at the last timestep
        return torch.argmax(q_values[:, -1, :]).item(), False

    def context_reset(self, obs: np.ndarray) -> None:
        """Reset context and Bag
        
        Args:
            obs: Initial observation
        """
        self.context.reset(obs)
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store_first_obs(obs)
        if self.bag.size > 0:
            self.bag.reset()

    def observe(self, obs: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an observation to the context. If the context would evict an observation to make room,
        try to put the observation in the bag, which may require evicting something else from the bag.
        
        If we are in training mode, we also add the transition to the replay buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Whether episode is done
        """
        # Ensure action is within valid range
        action = min(max(action, 0), self.num_actions - 1)
        
        evicted_obs, evicted_action = self.context.add_transition(
            obs, action, reward, done
        )
        # If there's an evicted observation, we need to decide if it should go into the bag
        if self.bag.size > 0 and evicted_obs is not None:
            # Ensure evicted action is within valid range
            if evicted_action is not None:
                evicted_action = min(max(evicted_action, 0), self.num_actions - 1)
            
            # Bag is full
            if not self.bag.add(evicted_obs, evicted_action):
                # For each possible bag, get Q values
                possible_bag_obss = np.tile(self.bag.obss, (self.bag.size + 1, 1, 1))
                possible_bag_actions = np.tile(self.bag.actions, (self.bag.size + 1, 1, 1))
                
                # Ensure all bag actions are within valid range
                possible_bag_actions = np.clip(possible_bag_actions, 0, self.num_actions - 1)
                
                for i in range(self.bag.size):
                    possible_bag_obss[i, i] = evicted_obs
                    possible_bag_actions[i, i] = evicted_action
                tiled_context = np.tile(self.context.obs, (self.bag.size + 1, 1, 1))
                tiled_actions = np.tile(self.context.action, (self.bag.size + 1, 1, 1))
                
                # Ensure all context actions are within valid range
                tiled_actions = np.clip(tiled_actions, 0, self.num_actions - 1)
                
                q_values = self.policy_network(
                    torch.as_tensor(
                        tiled_context,
                        dtype=self.obs_tensor_type,
                        device=self.device
                    ),
                    torch.as_tensor(
                        tiled_actions,
                        dtype=torch.long,
                        device=self.device
                    ),
                    torch.as_tensor(
                        possible_bag_obss,
                        dtype=self.obs_tensor_type,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        possible_bag_actions,
                        dtype=torch.long,
                        device=self.device
                    ),
                )

                bag_idx = torch.argmax(torch.mean(torch.max(q_values, 2)[0], 1))
                self.bag.obss = possible_bag_obss[bag_idx]
                self.bag.actions = possible_bag_actions[bag_idx]

        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store(obs, action, reward, done, self.context.timestep)

    def train(self) -> None:
        """Perform one training update, supporting prioritized experience replay"""
        if not self.replay_buffer.can_sample(self.batch_size):
            return
            
        self.eval_off()
        
        # Initialize variables for PER outputs
        weights_tensor = torch.ones(self.batch_size, device=self.device, dtype=torch.float32)
        sampled_episode_indices_for_update = None
        sampled_tree_node_indices_for_update = None

        if self.bag.size > 0:
            sample_results = self.replay_buffer.sample_with_bag(self.batch_size, self.bag)
            if sample_results is None: return # Sampling failed

            if hasattr(self.replay_buffer, 'update_priorities'): 
                (
                    obss_np, actions_np, rewards_np, next_obss_np, next_actions_np, dones_np,
                    episode_lengths_np, weights_np,
                    sampled_episode_indices_for_update, sampled_tree_node_indices_for_update,
                    # sampled_context_start_indices, # This is returned by sample_with_bag but not directly used by agent.train
                    bag_obss_np, bag_actions_np 
                ) = sample_results
                weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=self.device)
            else:
                (
                    obss_np, actions_np, rewards_np, next_obss_np, next_actions_np, dones_np,
                    episode_lengths_np, bag_obss_np, bag_actions_np
                ) = sample_results
            
            bag_actions_np = np.clip(bag_actions_np, 0, self.num_actions - 1)
            bag_obss_tensor = torch.as_tensor(bag_obss_np, dtype=self.obs_tensor_type, device=self.device)
            bag_actions_tensor = torch.as_tensor(bag_actions_np, dtype=torch.long, device=self.device)
        else:
            sample_results = self.replay_buffer.sample(self.batch_size)
            if sample_results is None: return # Sampling failed

            if hasattr(self.replay_buffer, 'update_priorities'):
                (
                    obss_np, actions_np, rewards_np, next_obss_np, next_actions_np, dones_np,
                    episode_lengths_np, weights_np,
                    sampled_episode_indices_for_update, sampled_tree_node_indices_for_update,
                    _ # context_start_indices returned but not used here
                ) = sample_results
                weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=self.device)
            else:
                (
                    obss_np, actions_np, rewards_np, next_obss_np, next_actions_np, dones_np,
                    episode_lengths_np
                ) = sample_results
            
            bag_obss_tensor = None
            bag_actions_tensor = None

        actions_np = np.clip(actions_np, 0, self.num_actions - 1)
        next_actions_np = np.clip(next_actions_np, 0, self.num_actions - 1)

        obss_tensor = torch.as_tensor(obss_np, dtype=self.obs_tensor_type, device=self.device)
        next_obss_tensor = torch.as_tensor(next_obss_np, dtype=self.obs_tensor_type, device=self.device)
        actions_tensor = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)
        next_actions_tensor = torch.as_tensor(next_actions_np, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones_np, dtype=torch.long, device=self.device)
        
        q_values = self.policy_network(obss_tensor, actions_tensor, bag_obss_tensor, bag_actions_tensor)
        q_values = q_values.gather(2, actions_tensor).squeeze()

        with torch.no_grad():
            if self.history: # Double DQN part
                argmax_actions = torch.argmax(
                    self.policy_network(next_obss_tensor, next_actions_tensor, bag_obss_tensor, bag_actions_tensor), dim=2
                ).unsqueeze(-1)
                next_obs_q_values = self.target_network(
                    next_obss_tensor, next_actions_tensor, bag_obss_tensor, bag_actions_tensor
                )
                next_obs_q_values = next_obs_q_values.gather(2, argmax_actions).squeeze()
            else: # Standard DQN (if history is False, though DTQN usually implies history)
                next_obs_q_values = self.target_network(next_obss_tensor, next_actions_tensor, bag_obss_tensor, bag_actions_tensor).max(dim=2)[0]
            
            targets = rewards_tensor.squeeze() + (1 - dones_tensor.squeeze()) * (next_obs_q_values * self.gamma)

        # Align history for loss calculation if context_len and history are different (usually same for DTQN)
        q_values_hist = q_values[:, -self.history:]
        targets_hist = targets[:, -self.history:]
        
        td_errors_per_step = torch.abs(q_values_hist - targets_hist)
        
        # For PER, often a single TD error per sampled transition (context) is used.
        # Here, td_errors_per_step is [batch_size, history_len]. 
        # The PERBuffer.update_priorities now expects [batch_size] or handles averaging.
        # Let's pass the [batch_size, history_len] errors directly, PERBuffer will average.
        td_errors_for_update = td_errors_per_step.detach().cpu().numpy()

        batch_weights_expanded = weights_tensor.unsqueeze(1).expand_as(q_values_hist)
        loss = (batch_weights_expanded * ((q_values_hist - targets_hist) ** 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        
        if hasattr(self.replay_buffer, 'update_priorities') and sampled_episode_indices_for_update is not None:
            self.replay_buffer.update_priorities(
                sampled_episode_indices_for_update, 
                sampled_tree_node_indices_for_update, 
                td_errors_for_update # Pass the [batch, history_len] errors
            )
        
        self.num_train_steps += 1

        # After each actual training step, advance beta annealing if using PER
        if hasattr(self.replay_buffer, 'step_beta_annealing'):
            self.replay_buffer.step_beta_annealing()

        self.td_errors.add(td_errors_per_step.mean().item()) # Log average TD error per step
        self.grad_norms.add(grad_norm.item())
        self.qvalue_max.add(q_values_hist.max().item())
        self.qvalue_mean.add(q_values_hist.mean().item())
        self.qvalue_min.add(q_values_hist.min().item())
        self.target_max.add(targets_hist.max().item())
        self.target_mean.add(targets_hist.mean().item())
        self.target_min.add(targets_hist.min().item())
        
        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()

    def target_update(self) -> None:
        """Hard update: copy parameters from policy network to target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_mini_checkpoint(self, checkpoint_dir: str, wandb_id: str) -> None:
        """Save mini checkpoint
        
        Args:
            checkpoint_dir: Checkpoint directory
            wandb_id: Wandb run ID
        """
        torch.save(
            {"step": self.num_train_steps, "wandb_id": wandb_id},
            checkpoint_dir + "_mini_checkpoint.pt",
        )

    @staticmethod
    def load_mini_checkpoint(checkpoint_dir: str) -> dict:
        """Load mini checkpoint
        
        Args:
            checkpoint_dir: Checkpoint directory
            
        Returns:
            dict: Checkpoint data
        """
        return torch.load(checkpoint_dir + "_mini_checkpoint.pt")

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        wandb_id: str,
        episode_successes: RunningAverage,
        episode_rewards: RunningAverage,
        episode_lengths: RunningAverage,
        eps: LinearAnneal,
    ) -> None:
        """Save full checkpoint
        
        Args:
            checkpoint_dir: Checkpoint directory
            wandb_id: Wandb run ID
            episode_successes: Success rate statistics
            episode_rewards: Reward statistics
            episode_lengths: Length statistics
            eps: Epsilon decay object
        """
        self.save_mini_checkpoint(checkpoint_dir=checkpoint_dir, wandb_id=wandb_id)
        torch.save(
            {
                "step": self.num_train_steps,
                "wandb_id": wandb_id,
                # Replay Buffer: Don't keep the observation index saved
                "replay_buffer_pos": [self.replay_buffer.pos[0], 0],
                # Neural Net
                "policy_net_state_dict": self.policy_network.state_dict(),
                "target_net_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": eps.val,
                # Results
                "episode_successes": episode_successes,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                # Losses
                "td_errors": self.td_errors,
                "grad_norms": self.grad_norms,
                "qvalue_max": self.qvalue_max,
                "qvalue_mean": self.qvalue_mean,
                "qvalue_min": self.qvalue_min,
                "target_max": self.target_max,
                "target_mean": self.target_mean,
                "target_min": self.target_min,
                # RNG states
                "random_rng_state": random.getstate(),
                "rng_bit_generator_state": RNG.rng.bit_generator.state,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else torch.get_rng_state(),
            },
            checkpoint_dir + "_checkpoint.pt",
        )
        joblib.dump(self.replay_buffer.obss, checkpoint_dir + "buffer_obss.sav")
        joblib.dump(self.replay_buffer.actions, checkpoint_dir + "buffer_actions.sav")
        joblib.dump(self.replay_buffer.rewards, checkpoint_dir + "buffer_rewards.sav")
        joblib.dump(self.replay_buffer.dones, checkpoint_dir + "buffer_dones.sav")
        joblib.dump(
            self.replay_buffer.episode_lengths, checkpoint_dir + "buffer_eplens.sav"
        )

    def load_checkpoint(
        self, checkpoint_dir: str
    ) -> Tuple[str, RunningAverage, RunningAverage, RunningAverage, float]:
        """Load checkpoint
        
        Args:
            checkpoint_dir: Checkpoint directory
            
        Returns:
            Tuple: (wandb_id, episode_successes, episode_rewards, episode_lengths, epsilon)
        """
        checkpoint = torch.load(checkpoint_dir + "_checkpoint.pt")

        self.num_train_steps = checkpoint["step"]
        # Replay Buffer
        self.replay_buffer.pos = checkpoint["replay_buffer_pos"]
        self.replay_buffer.obss = joblib.load(checkpoint_dir + "buffer_obss.sav")
        self.replay_buffer.actions = joblib.load(checkpoint_dir + "buffer_actions.sav")
        self.replay_buffer.rewards = joblib.load(checkpoint_dir + "buffer_rewards.sav")
        self.replay_buffer.dones = joblib.load(checkpoint_dir + "buffer_dones.sav")
        self.replay_buffer.episode_lengths = joblib.load(
            checkpoint_dir + "buffer_eplens.sav"
        )
        # Neural Net
        self.policy_network.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Losses
        self.td_errors = checkpoint["td_errors"]
        self.grad_norms = checkpoint["grad_norms"]
        self.qvalue_max = checkpoint["qvalue_max"]
        self.qvalue_mean = checkpoint["qvalue_mean"]
        self.qvalue_min = checkpoint["qvalue_min"]
        self.target_max = checkpoint["target_max"]
        self.target_mean = checkpoint["target_mean"]
        self.target_min = checkpoint["target_min"]
        # RNG states
        random.setstate(checkpoint["random_rng_state"])
        RNG.rng.bit_generator.state = checkpoint["rng_bit_generator_state"]
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["torch_cuda_rng_state"])

        # Results
        episode_successes = checkpoint["episode_successes"]
        episode_rewards = checkpoint["episode_rewards"]
        episode_lengths = checkpoint["episode_lengths"]
        # Exploration value
        epsilon = checkpoint["epsilon"]

        return (
            checkpoint["wandb_id"],
            episode_successes,
            episode_rewards,
            episode_lengths,
            epsilon,
        )
