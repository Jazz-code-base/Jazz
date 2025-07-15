import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from collections import deque

# Define a larger history length limit for heavyweight data
# You can adjust this value based on memory budget
# For example, 100k steps of states (16*4 bytes) is about 6MB, attention weights may be larger
VISUALIZER_MAX_HISTORY_LEN = 500000 # For example, 500k steps

class DTQNVisualizer:
    """DTQN visualization tool for analyzing and visualizing DTQN model decision processes"""
    
    def __init__(self, dqn_agent):
        """
        Initialize visualization tool
        
        Args:
            dqn_agent: DTQN agent instance
        """
        self.dqn_agent = dqn_agent
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set chart style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Record historical data
        self.history = {
            'rewards': [], # Keep as list, clear after export
            'sf1_rewards': [],  # Keep as list, clear after export
            'sf2_rewards': [],  # Keep as list, clear after export
            'actions': [], # Keep as list, clear after export
            'states': deque(maxlen=VISUALIZER_MAX_HISTORY_LEN), # Changed to deque
            'q_values': deque(maxlen=VISUALIZER_MAX_HISTORY_LEN), # q_values can also be large
            'losses': [], # Usually small, keep as list, clear after export (if needed for charts or CSV)
            'attn_weights': deque(maxlen=VISUALIZER_MAX_HISTORY_LEN), # Changed to deque (bag attention)
            'epsilon': [], # Keep as list, clear after export
            'learning_rates': [],  # Keep as list, clear after export
            'elapsed_times': []  # Keep as list, clear after export
            # transformer_attn_weights will be dynamically initialized as deque list in record_step
        }
        self.total_steps_recorded_for_list_history = 0 # For tracking current data volume in non-deque lists
    
    def save_attention_weights(self, step_idx=None, suffix=""):
        """Save attention weights heatmap
        
        Args:
            step_idx: Step index, if None uses latest data
            suffix: Filename suffix
        """
        # Check if bag mechanism is enabled
        if hasattr(self.dqn_agent.dtqn_agent, 'bag') and self.dqn_agent.dtqn_agent.bag_size > 0:
            # When using bag mechanism, save bag attention weights
            # Note: attn_weights here is instantaneous, might need to get from self.history['attn_weights'] for specific step
            # For simplicity, we assume policy_network.attn_weights is the latest available for saving
            attn_weights_data = self.dqn_agent.dtqn_agent.policy_network.attn_weights
            if attn_weights_data is None: # If instantaneous is empty, try to get the last one from history
                if len(self.history['attn_weights']) > 0:
                    attn_weights_data = self.history['attn_weights'][-1] 
                else:
                    print("Bag attention weights empty (both instantaneous and history), trying to save transformer attention weights")
                    self.save_transformer_attention_weights(step_idx, suffix)
                    return
            
            weights = attn_weights_data.cpu().detach().numpy() if torch.is_tensor(attn_weights_data) else attn_weights_data
            if len(weights.shape) > 2: 
                weights = weights.mean(axis=0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights, annot=False, cmap="viridis")
            plt.title("Bag Attention Weights Heatmap")
            plt.xlabel("Bag Position")
            plt.ylabel("Query Position")
            
            filename_stem = "bag_attention_weights"
            if step_idx is not None:
                file_name = f"{filename_stem}_{step_idx}{suffix}.png"
            else:
                file_name = f"{filename_stem}_latest{suffix}.png"
            
            plt.savefig(self.output_dir / file_name, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Bag attention weights heatmap saved to {self.output_dir / file_name}")
        
        self.save_transformer_attention_weights(step_idx, suffix)
    
    def visualize_bag_contents(self, step_idx=None, suffix=""):
        """Visualize contents in the Bag
        
        Args:
            step_idx: Step index, if None uses latest data
            suffix: Filename suffix
        """
        if not hasattr(self.dqn_agent.dtqn_agent, 'bag') or self.dqn_agent.dtqn_agent.bag_size == 0:
            print("Bag mechanism disabled, cannot visualize Bag contents")
            return
            
        bag = self.dqn_agent.dtqn_agent.bag
        
        if bag.pos == 0: # bag.pos refers to the number of valid elements in the bag
            print("Bag is empty, cannot visualize")
            return
        
        bag_obss = bag.obss[:bag.pos]
        bag_actions = bag.actions[:bag.pos]
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), squeeze=False) # squeeze=False ensures axs is always 2D
        
        if len(bag_obss) > 0:
            sns.heatmap(bag_obss, ax=axs[0,0], cmap="viridis")
            axs[0,0].set_title("Observations in Bag")
            axs[0,0].set_xlabel("Observation Dimension")
            axs[0,0].set_ylabel("Bag Sequence Index")
        
        if len(bag_actions) > 0:
            axs[1,0].bar(range(len(bag_actions)), bag_actions.flatten())
            axs[1,0].set_title("Actions in Bag")
            axs[1,0].set_xlabel("Bag Sequence Index")
            axs[1,0].set_ylabel("Action ID")
            axs[1,0].set_ylim(0, self.dqn_agent.n_actions)
            
        plt.tight_layout()
        
        filename_stem = "bag_contents"
        if step_idx is not None:
            file_name = f"{filename_stem}_{step_idx}{suffix}.png"
        else:
            file_name = f"{filename_stem}_latest{suffix}.png"
        
        plt.savefig(self.output_dir / file_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Bag contents visualization saved to {self.output_dir / file_name}")
    
    def record_step(self, state, action, reward, q_values=None, epsilon=None, sf1_reward=None, sf2_reward=None, elapsed_time=None):
        """Record data for one step
        
        Args:
            state: Current state
            action: Selected action
            reward: Received reward
            q_values: Q-values (optional)
            epsilon: Exploration rate (optional)
            sf1_reward: Subflow 1 reward (optional)
            sf2_reward: Subflow 2 reward (optional)
            elapsed_time: Execution time (optional)
        """
        # For non-deque lists, append normally
        self.history['rewards'].append(reward)
        self.history['actions'].append(action)
        if sf1_reward is not None:
            self.history['sf1_rewards'].append(sf1_reward)
        if sf2_reward is not None:
            self.history['sf2_rewards'].append(sf2_reward)
        if epsilon is not None:
            self.history['epsilon'].append(epsilon)
        if hasattr(self.dqn_agent, 'optimizer') and self.dqn_agent.optimizer and self.dqn_agent.optimizer.param_groups:
            self.history['learning_rates'].append(
                self.dqn_agent.optimizer.param_groups[0]['lr']
            )
        else:
            self.history['learning_rates'].append(None) # Or a marker value

        if elapsed_time is not None:
            self.history['elapsed_times'].append(elapsed_time)
        
        # For deque lists, append normally, they will automatically manage length
        self.history['states'].append(state.cpu().numpy() if torch.is_tensor(state) else state)
        if q_values is not None:
            self.history['q_values'].append(q_values.cpu().numpy() if torch.is_tensor(q_values) else q_values)

        if (hasattr(self.dqn_agent.dtqn_agent, 'policy_network') and 
            hasattr(self.dqn_agent.dtqn_agent.policy_network, 'attn_weights') and 
            self.dqn_agent.dtqn_agent.policy_network.attn_weights is not None):
            self.history['attn_weights'].append(
                self.dqn_agent.dtqn_agent.policy_network.attn_weights.cpu().detach().numpy()
            )
        
        if hasattr(self.dqn_agent.dtqn_agent, 'policy_network') and hasattr(self.dqn_agent.dtqn_agent.policy_network, 'transformer_layers'):
            transformer_layers = self.dqn_agent.dtqn_agent.policy_network.transformer_layers
            
            if 'transformer_attn_weights' not in self.history or not isinstance(self.history['transformer_attn_weights'], list):
                self.history['transformer_attn_weights'] = [deque(maxlen=VISUALIZER_MAX_HISTORY_LEN) for _ in range(len(transformer_layers))]
                
            for i, layer in enumerate(transformer_layers):
                if hasattr(layer, 'alpha') and layer.alpha is not None:
                    if i < len(self.history['transformer_attn_weights']): # Ensure index is valid
                        self.history['transformer_attn_weights'][i].append(
                            layer.alpha.cpu().detach().numpy()
                        )
        self.total_steps_recorded_for_list_history += 1
    
    def record_transformer_attention_only(self, state):
        """Only record transformer attention weights in minimalist mode
        
        Args:
            state: Current state (for model inference to get attention weights)
        """
        # Note: sample_action should have already called policy_network, so transformer attention weights should be updated
        
        # Record transformer attention weights
        if hasattr(self.dqn_agent.dtqn_agent, 'policy_network') and hasattr(self.dqn_agent.dtqn_agent.policy_network, 'transformer_layers'):
            transformer_layers = self.dqn_agent.dtqn_agent.policy_network.transformer_layers
            
            if 'transformer_attn_weights' not in self.history or not isinstance(self.history['transformer_attn_weights'], list):
                self.history['transformer_attn_weights'] = [deque(maxlen=VISUALIZER_MAX_HISTORY_LEN) for _ in range(len(transformer_layers))]
                
            for i, layer in enumerate(transformer_layers):
                if hasattr(layer, 'alpha') and layer.alpha is not None:
                    if i < len(self.history['transformer_attn_weights']): # Ensure index is valid
                        self.history['transformer_attn_weights'][i].append(
                            layer.alpha.cpu().detach().numpy()
                        )
    
    def plot_reward_history(self, window_size=100, save=True):
        """Plot reward history
        
        Args:
            window_size: Moving average window size
            save: Whether to save the chart
        """
        # Use self.history['rewards'] (this is a list)
        rewards_to_plot = list(self.history['rewards']) # Create a copy to prevent modification during iteration (though risk is low here)
        if not rewards_to_plot:
            print("No reward history data for plotting")
            return
        
        x = range(len(rewards_to_plot))
        
        fig, ax = plt.subplots(figsize=(10, 6)) # Single chart
        ax.plot(x, rewards_to_plot, alpha=0.3, label='Original Reward')
        
        if len(rewards_to_plot) >= window_size:
            moving_avg = pd.Series(rewards_to_plot).rolling(window=window_size).mean().iloc[window_size-1:].values
            ax.plot(range(window_size-1, len(rewards_to_plot)), moving_avg, label=f'Moving Avg (Window={window_size})')
        
        ax.set_title('Reward History (Current Interval)')
        ax.set_xlabel('Steps in Current Interval')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / "reward_history.png", dpi=300, bbox_inches="tight")
            print(f"Reward history chart saved to {self.output_dir / 'reward_history.png'}")
        
        plt.close(fig)
    
    def plot_action_distribution(self, save=True):
        """Plot action distribution
        
        Args:
            save: Whether to save the chart
        """
        actions_to_plot = list(self.history['actions'])
        if not actions_to_plot:
            print("No action history data for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        action_counts = np.bincount(actions_to_plot, minlength=self.dqn_agent.n_actions)
        ax.bar(range(self.dqn_agent.n_actions), action_counts)
        ax.set_title('Action Distribution (Current Interval)')
        ax.set_xlabel('Action ID')
        ax.set_ylabel('Action Count')
        ax.set_xticks(range(self.dqn_agent.n_actions))
        ax.grid(True, axis='y')
        
        if save:
            plt.savefig(self.output_dir / "action_distribution.png", dpi=300, bbox_inches="tight")
            print(f"Action distribution chart saved to {self.output_dir / 'action_distribution.png'}")
        
        plt.close(fig)
    
    def plot_epsilon_decay(self, save=True):
        """Plot epsilon decay curve
        
        Args:
            save: Whether to save the chart
        """
        epsilons_to_plot = list(self.history['epsilon'])
        if not epsilons_to_plot:
            print("No epsilon history data for plotting")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(epsilons_to_plot)), epsilons_to_plot)
        ax.set_title('Epsilon Decay (Current Interval)')
        ax.set_xlabel('Steps in Current Interval')
        ax.set_ylabel('Epsilon Value')
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / "epsilon_decay.png", dpi=300, bbox_inches="tight")
            print(f"Epsilon decay curve saved to {self.output_dir / 'epsilon_decay.png'}")
            
        plt.close(fig)
    
    def generate_analysis_report(self, save=True):
        """Generate analysis report (based on current list data in memory)
        
        Args:
            save: Whether to save the report
        """
        try:
            rewards_current_interval = list(self.history['rewards'])
            if not rewards_current_interval:
                print("Not enough historical data in the current interval to generate a report")
                return
            
            base_length = len(rewards_current_interval)
            
            actions_current_interval = list(self.history['actions'])
            most_common_action = None
            if actions_current_interval and len(actions_current_interval) == base_length:
                most_common_action = np.bincount(actions_current_interval).argmax()
            
            epsilon_current_interval = list(self.history['epsilon'])
            latest_epsilon = None
            if epsilon_current_interval and len(epsilon_current_interval) > 0: # Check not empty
                latest_epsilon = epsilon_current_interval[-1]
            
            report = {
                'Steps in Current Interval': base_length,
                'Average Reward (Current Interval)': np.mean(rewards_current_interval) if base_length > 0 else None,
                'Maximum Reward (Current Interval)': np.max(rewards_current_interval) if base_length > 0 else None,
                'Minimum Reward (Current Interval)': np.min(rewards_current_interval) if base_length > 0 else None,
                'Most Common Action (Current Interval)': most_common_action,
                'Final Epsilon (Current Interval)': latest_epsilon
            }
            
            print("--- Analysis Report (Current Interval) ---")
            for key, value in report.items():
                print(f"{key}: {value}")
            
            if save:
                with open(self.output_dir / "analysis_report_current_interval.txt", "w") as f:
                    for key, value in report.items():
                        f.write(f"{key}: {value}\\n")
                print(f"Current interval analysis report saved to {self.output_dir / 'analysis_report_current_interval.txt'}")
            
            if 'transformer_attn_weights' in self.history and isinstance(self.history['transformer_attn_weights'], list):
                num_layers = len(self.history['transformer_attn_weights'])
                for layer_idx in range(num_layers):
                    if self.history['transformer_attn_weights'][layer_idx]: # Check if deque is empty
                        try:
                            self.plot_attention_evolution(layer_idx=layer_idx, save=True)
                            print(f"Transformer layer {layer_idx} attention weights evolution chart generated (based on current deque)")
                        except Exception as e:
                            print(f"Error generating transformer layer {layer_idx} attention weights evolution chart: {e}")
            
            return report
        except Exception as e:
            print(f"Error generating analysis report: {e}")
            return None
    
    def export_history_to_csv(self):
        """Export historical data to CSV files.
        For all data types, only export a snapshot of the current data, do not clear the data.
        """
        try:
            # Process list type data
            list_data_to_export = {}
            base_length = self.total_steps_recorded_for_list_history # Current interval recorded steps
            
            if base_length == 0:
                print("No list type data to export to training_history.csv in the current interval")
            else:
                list_data_to_export['step_in_interval'] = list(range(base_length))
                
                list_keys = ['rewards', 'sf1_rewards', 'sf2_rewards', 'actions', 'epsilon', 'learning_rates', 'elapsed_times', 'losses']
                for key in list_keys:
                    if key in self.history and isinstance(self.history[key], list):
                        current_list = self.history[key]
                        # Ensure exported data length matches base_length
                        if len(current_list) >= base_length:
                            list_data_to_export[key] = current_list[:base_length]
                        else: # If list length is insufficient
                            list_data_to_export[key] = current_list + [None] * (base_length - len(current_list))
                    elif key not in self.history: # If a key is not initialized
                        list_data_to_export[key] = [None] * base_length
                
                df_training = pd.DataFrame(list_data_to_export)
                # Overwrite each time, only include data from the current interval
                training_csv_path = self.output_dir / "training_history_interval.csv"
                df_training.to_csv(training_csv_path, index=False)
                print(f"Training history data for the current interval exported to {training_csv_path}")

            # Process deque type states data
            if len(self.history['states']) > 0:
                try:
                    states_list = list(self.history['states']) # Convert deque to list
                    states_np = np.array(states_list)
                    
                    if states_np.ndim > 1 and states_np.shape[0] > 0:
                        num_dims_to_save = min(8, states_np.shape[-1])
                        # Ensure states_np is 2D (batch, features)
                        if states_np.ndim > 2: # For example (batch, seq_len, features) -> (batch*seq_len, features)
                             states_reshaped = states_np.reshape(-1, states_np.shape[-1])
                        else:
                             states_reshaped = states_np

                        states_df_data = states_reshaped[:, :num_dims_to_save]
                        
                        states_df = pd.DataFrame(
                            states_df_data,
                            columns=[f'state_dim_{i}' for i in range(num_dims_to_save)]
                        )
                        state_csv_path = self.output_dir / "state_history_deque_snapshot.csv"
                        states_df.to_csv(state_csv_path, index=False)
                        print(f"State history (deque snapshot) exported to {state_csv_path}")
                except Exception as e:
                    print(f"Error exporting state history (deque): {e}")
            else:
                print("States deque is empty, not exporting state_history_deque_snapshot.csv")

            # Export Q-value data
            if len(self.history['q_values']) > 0:
                try:
                    q_values_list = list(self.history['q_values'])
                    q_values_np = np.array(q_values_list)
                    # Assume q_values_np is (N, num_actions)
                    if q_values_np.ndim == 2 and q_values_np.shape[0] > 0:
                         q_values_df = pd.DataFrame(q_values_np, columns=[f'q_action_{i}' for i in range(q_values_np.shape[1])])
                         q_values_csv_path = self.output_dir / "q_values_deque_snapshot.csv"
                         q_values_df.to_csv(q_values_csv_path, index=False)
                         print(f"Q-values (deque snapshot) exported to {q_values_csv_path}")
                except Exception as e:
                    print(f"Error exporting Q-values (deque): {e}")

        except Exception as e:
            print(f"A severe error occurred while exporting CSV historical data: {e}")

    def save_transformer_attention_weights(self, step_idx=None, suffix=""):
        """Save heatmap of attention weights for Transformer layers, showing weights between time steps
        
        Args:
            step_idx: Step index, if None uses latest data
            suffix: Filename suffix
        """
        # First try to get the latest attention weights directly from transformer layers
        if (hasattr(self.dqn_agent.dtqn_agent, 'policy_network') and 
            hasattr(self.dqn_agent.dtqn_agent.policy_network, 'transformer_layers')):
            
            transformer_layers = self.dqn_agent.dtqn_agent.policy_network.transformer_layers
            
            for i, layer in enumerate(transformer_layers):
                if hasattr(layer, 'alpha') and layer.alpha is not None:
                    # Use the latest attention weights directly from the transformer layer
                    weights_data = layer.alpha
                    weights = weights_data.cpu().detach().numpy() if torch.is_tensor(weights_data) else weights_data
                    
                    if len(weights.shape) > 2:
                        weights = weights.mean(axis=0) # Multi-head average
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(weights, annot=False, cmap="viridis", ax=ax)
                    
                    ax.set_title(f"Transformer Layer {i} Attention Weights (Latest)")
                    ax.set_xlabel("Key Position (Historical Time Steps)")
                    ax.set_ylabel("Query Position (Current Time Steps)")
                    
                    history_len_viz = weights.shape[0]
                    fig.text(0.5, 0.01, 
                                f"Y-axis: Query Position of Current Time Step (0-{history_len_viz-1})\\n"
                                f"X-axis: Key Position of Historical Time Steps (0-{history_len_viz-1})\\n"
                                f"Darker colors indicate higher attention weights",
                                ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
                    
                    filename_stem = f"transformer_attention_weights_layer{i}"
                    if step_idx is not None:
                        file_name = f"{filename_stem}_globalStep{step_idx}{suffix}.png"
                    else:
                        file_name = f"{filename_stem}_latest{suffix}.png"
                    
                    plt.savefig(self.output_dir / file_name, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Transformer layer {i} attention weights heatmap saved to {self.output_dir / file_name}")
                else:
                    print(f"Transformer layer {i} has no available attention weights (alpha is None)")
            
            return  # Return directly after successfully getting weights from transformer layers
        
        # If unable to get directly, fall back to using history (for full visualization mode)
        if 'transformer_attn_weights' not in self.history or not isinstance(self.history['transformer_attn_weights'], list):
            print("Transformer attention weights history not initialized or format incorrect, and cannot be directly obtained from transformer layers")
            return

        for i, layer_history_deque in enumerate(self.history['transformer_attn_weights']):
            if not layer_history_deque: # If deque is empty
                print(f"Attention weights history (deque) for Transformer layer {i} is empty, cannot generate heatmap")
                continue

            # Get the last (latest) attention weights from the deque
            weights_data = layer_history_deque[-1]
            weights = weights_data.cpu().detach().numpy() if torch.is_tensor(weights_data) else weights_data
            
            if len(weights.shape) > 2:
                weights = weights.mean(axis=0) # Multi-head average
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(weights, annot=False, cmap="viridis", ax=ax)
            
            ax.set_title(f"Transformer Layer {i} Attention Weights (Latest in Deque)")
            ax.set_xlabel("Key Position (Historical Time Steps)")
            ax.set_ylabel("Query Position (Current Time Steps)")
            
            history_len_viz = weights.shape[0]
            fig.text(0.5, 0.01, 
                        f"Y-axis: Query Position of Current Time Step (0-{history_len_viz-1})\\n"
                        f"X-axis: Key Position of Historical Time Steps (0-{history_len_viz-1})\\n"
                        f"Darker colors indicate higher attention weights",
                        ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            filename_stem = f"transformer_attention_weights_layer{i}"
            if step_idx is not None: # step_idx here might refer to global training steps
                file_name = f"{filename_stem}_globalStep{step_idx}{suffix}.png"
            else:
                file_name = f"{filename_stem}_latest_in_deque{suffix}.png"
            
            plt.savefig(self.output_dir / file_name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Transformer layer {i} attention weights heatmap saved to {self.output_dir / file_name}")
            
    def plot_attention_evolution(self, layer_idx=0, num_snapshots=10, save=True):
        """Plot the evolution of transformer self-attention weights over time steps (based on snapshots in deque)
        
        Args:
            layer_idx: Index of the transformer layer to visualize
            num_snapshots: Number of snapshots to visualize
            save: Whether to save the chart
        """
        if 'transformer_attn_weights' not in self.history or not isinstance(self.history['transformer_attn_weights'], list) \
           or len(self.history['transformer_attn_weights']) <= layer_idx:
            print(f"No attention weights history for transformer layer {layer_idx}")
            return
            
        layer_history_deque = self.history['transformer_attn_weights'][layer_idx]
        if not layer_history_deque:
            print(f"Attention weights history (deque) for Transformer layer {layer_idx} is empty")
            return
            
        deque_len = len(layer_history_deque)
        if deque_len < num_snapshots:
            print(f"Not enough data in deque ({deque_len} snapshots) to display fewer than requested {num_snapshots} snapshots. All available snapshots will be displayed.")
            indices_to_plot = range(deque_len)
            actual_num_snapshots = deque_len
        else:
            # Uniformly sample snapshot indices from the deque
            indices_to_plot = np.linspace(0, deque_len - 1, num_snapshots, dtype=int)
            actual_num_snapshots = num_snapshots

        if actual_num_snapshots == 0:
            print(f"No snapshots available for plotting Transformer layer {layer_idx}.")
            return

        grid_size = int(np.ceil(np.sqrt(actual_num_snapshots)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4), squeeze=False)
        fig.suptitle(f"Transformer Layer {layer_idx} Attention Weights Evolution (Snapshots from Deque)", fontsize=16)
        
        axes_flat = axes.flatten()
        
        for i, snapshot_idx_in_deque in enumerate(indices_to_plot):
            if i >= len(axes_flat): break
                
            ax = axes_flat[i]
            weights_data = layer_history_deque[snapshot_idx_in_deque]
            weights = weights_data.cpu().detach().numpy() if torch.is_tensor(weights_data) else weights_data
            
            if len(weights.shape) > 2: # Multi-head average
                weights = weights.mean(axis=0)
                
            sns.heatmap(weights, annot=False, cmap="viridis", ax=ax)
            ax.set_title(f"Snapshot {snapshot_idx_in_deque+1}/{deque_len}") # Display in deque position
            
            is_last_row = (i // grid_size == grid_size - 1) or (i >= actual_num_snapshots - (actual_num_snapshots % grid_size if actual_num_snapshots % grid_size !=0 else grid_size) )
            is_first_col = (i % grid_size == 0)

            ax.set_xlabel("Key Position" if is_last_row else "")
            ax.set_ylabel("Query Position" if is_first_col else "")
            ax.tick_params(labelbottom=is_last_row, labelleft=is_first_col)

        for k in range(i + 1, len(axes_flat)):
            axes_flat[k].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save:
            file_name = f"attention_evolution_layer{layer_idx}_deque.png"
            plt.savefig(self.output_dir / file_name, dpi=300, bbox_inches="tight")
            print(f"Attention weights evolution chart (deque) saved to {self.output_dir / file_name}")
            
        plt.close(fig)


def compare_models(old_model_path, new_model_path, dqn_agent, test_episodes=10):
    """Compare performance of two models
    
    Args:
        old_model_path: Path to old model
        new_model_path: Path to new model
        dqn_agent: DTQN agent instance
        test_episodes: Number of test episodes
    
    Returns:
        dict: Dictionary containing comparison results
    """
    results = {'old_model': {'rewards': []}, 'new_model': {'rewards': []}}
    
    # Create test state
    def create_test_state():
        # Subflow 1 state
        subflow1_state = np.array([
            np.random.uniform(0, 100),    # Throughput
            np.random.uniform(0, 2000),   # RTT
            np.random.uniform(0, 1),      # Packet loss rate
            np.random.uniform(10, 100),   # Congestion window
            np.random.uniform(0, 50),     # Last window change
            np.random.uniform(0, 1),      # Force decrease flag
            np.random.uniform(0, 1000),   # Last increase steps
            np.random.uniform(0, 1)       # Reserved
        ])
        
        # Subflow 2 state
        subflow2_state = np.array([
            np.random.uniform(0, 100),
            np.random.uniform(0, 2000),
            np.random.uniform(0, 1),
            np.random.uniform(10, 100),
            np.random.uniform(0, 50),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1000),
            np.random.uniform(0, 1)
        ])
        
        # Combine states
        combined_state = np.concatenate([subflow1_state, subflow2_state])
        return torch.tensor(combined_state, dtype=torch.float32, device=dqn_agent.device)
    
    # Set random seed for fair comparison
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test old model
    if os.path.exists(old_model_path):
        dqn_agent.load_model(old_model_path)
        print(f"Old model loaded: {old_model_path}")
        
        for ep in range(test_episodes):
            episode_reward = 0
            dqn_agent.prev_state = None
            
            for step in range(20):  # 20 steps per test sequence
                state = create_test_state()
                action = dqn_agent.predict_action(state)  # Use deterministic policy
                reward = torch.tensor(np.random.uniform(-1, 1), device=dqn_agent.device)
                next_state = create_test_state()
                done = torch.tensor(0.0, device=dqn_agent.device) if step < 19 else torch.tensor(1.0, device=dqn_agent.device)
                
                dqn_agent.observe(state, action, reward, next_state, done)
                episode_reward += reward.item()
            
            results['old_model']['rewards'].append(episode_reward)
        
        results['old_model']['avg_reward'] = np.mean(results['old_model']['rewards'])
        print(f"Average reward for old model: {results['old_model']['avg_reward']:.4f}")
    else:
        print(f"Old model file does not exist: {old_model_path}")
    
    # Test new model
    if os.path.exists(new_model_path):
        dqn_agent.load_model(new_model_path)
        print(f"New model loaded: {new_model_path}")
        
        for ep in range(test_episodes):
            episode_reward = 0
            dqn_agent.prev_state = None
            
            for step in range(20):  # 20 steps per test sequence
                state = create_test_state()
                action = dqn_agent.predict_action(state)  # Use deterministic policy
                reward = torch.tensor(np.random.uniform(-1, 1), device=dqn_agent.device)
                next_state = create_test_state()
                done = torch.tensor(0.0, device=dqn_agent.device) if step < 19 else torch.tensor(1.0, device=dqn_agent.device)
                
                dqn_agent.observe(state, action, reward, next_state, done)
                episode_reward += reward.item()
            
            results['new_model']['rewards'].append(episode_reward)
        
        results['new_model']['avg_reward'] = np.mean(results['new_model']['rewards'])
        print(f"Average reward for new model: {results['new_model']['avg_reward']:.4f}")
    else:
        print(f"New model file does not exist: {new_model_path}")
    
    # Plot comparison chart
    if 'avg_reward' in results['old_model'] and 'avg_reward' in results['new_model']:
        plt.figure(figsize=(10, 6))
        
        plt.boxplot([results['old_model']['rewards'], results['new_model']['rewards']], 
                   labels=['Old Model', 'New Model'])
        
        plt.title('Model Performance Comparison')
        plt.ylabel('Cumulative Reward')
        plt.grid(True)
        
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Model comparison chart saved")
    
    return results


if __name__ == "__main__":
    print("DTQN Visualization Tool")
    print("Please import and use this module in server.py") 