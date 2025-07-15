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
        rewards_to_plot = list(self.history['rewards']) # Create copy to prevent modification during iteration (though risk is low here)
        
        if len(rewards_to_plot) == 0:
            print("No reward history data available for plotting")
            return
        
        # Calculate moving average
        moving_avg = pd.Series(rewards_to_plot).rolling(window=min(window_size, len(rewards_to_plot))).mean()
        
        fig, ax = plt.subplots(figsize=(10, 6)) # Single chart
        ax.plot(rewards_to_plot, alpha=0.3, label='Raw Rewards')
        ax.plot(moving_avg, label=f'Moving Avg (window={window_size})')
        ax.set_title('Reward History')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / 'reward_history.png', dpi=300, bbox_inches="tight")
            print(f"Reward history chart saved to {self.output_dir / 'reward_history.png'}")
        else:
            plt.show()
        plt.close()
    
    def plot_action_distribution(self, save=True):
        """Plot action distribution
        
        Args:
            save: Whether to save the chart
        """
        if len(self.history['actions']) == 0:
            print("No action history data available for plotting")
            return
        
        action_counts = np.bincount(self.history['actions'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(action_counts)), action_counts)
        ax.set_title('Action Distribution')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(action_counts)))
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / 'action_distribution.png', dpi=300, bbox_inches="tight")
            print(f"Action distribution chart saved to {self.output_dir / 'action_distribution.png'}")
        else:
            plt.show()
        plt.close()
    
    def plot_epsilon_decay(self, save=True):
        """Plot epsilon decay curve
        
        Args:
            save: Whether to save the chart
        """
        if len(self.history['epsilon']) == 0:
            print("No epsilon history data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history['epsilon'])
        ax.set_title('Epsilon Decay')
        ax.set_xlabel('Step')
        ax.set_ylabel('Epsilon')
        ax.grid(True)
        
        if save:
            plt.savefig(self.output_dir / 'epsilon_decay.png', dpi=300, bbox_inches="tight")
            print(f"Epsilon decay curve saved to {self.output_dir / 'epsilon_decay.png'}")
        else:
            plt.show()
        plt.close()
    
    def generate_analysis_report(self, save=True):
        """Generate analysis report (based on current in-memory list data)
        
        Args:
            save: Whether to save the report
        """
        try:
            if self.total_steps_recorded_for_list_history < 10:  # Arbitrary threshold
                print("Not enough historical data in current interval to generate report")
                return
                
            # Get current interval data
            rewards_current_interval = self.history['rewards']
            actions_current_interval = self.history['actions']
            epsilon_current_interval = self.history['epsilon']
            
            # Basic statistics
            report = []
            report.append("===== DTQN Analysis Report (Current Interval) =====")
            report.append(f"Total steps recorded: {self.total_steps_recorded_for_list_history}")
            
            # Reward statistics
            report.append("\n--- Reward Statistics ---")
            report.append(f"Average reward: {np.mean(rewards_current_interval):.4f}")
            report.append(f"Min reward: {np.min(rewards_current_interval):.4f}")
            report.append(f"Max reward: {np.max(rewards_current_interval):.4f}")
            report.append(f"Reward standard deviation: {np.std(rewards_current_interval):.4f}")
            
            # Action statistics
            report.append("\n--- Action Statistics ---")
            action_counts = np.bincount(actions_current_interval)
            for action, count in enumerate(action_counts):
                report.append(f"Action {action}: {count} times ({count/len(actions_current_interval)*100:.2f}%)")
            
            # Epsilon statistics
            report.append("\n--- Epsilon Statistics ---")
            if epsilon_current_interval and len(epsilon_current_interval) > 0: # Check not empty
                report.append(f"Starting epsilon: {epsilon_current_interval[0]:.4f}")
                report.append(f"Ending epsilon: {epsilon_current_interval[-1]:.4f}")
                report.append(f"Epsilon decay rate: {(epsilon_current_interval[0] - epsilon_current_interval[-1]) / len(epsilon_current_interval):.6f} per step")
            
            report_text = "\n".join(report)
            
            if save:
                with open(self.output_dir / 'analysis_report_current_interval.txt', 'w') as f:
                    f.write(report_text)
                print(f"Current interval analysis report saved to {self.output_dir / 'analysis_report_current_interval.txt'}")
            else:
                print(report_text)
                
            # Generate attention weight evolution chart for each layer
            for layer_idx in range(len(self.history.get('transformer_attn_weights', []))):
                try:
                    if self.history['transformer_attn_weights'][layer_idx]: # Check deque is not empty
                        self.plot_attention_evolution(layer_idx=layer_idx, save=save)
                        print(f"Generated transformer layer {layer_idx} attention weight evolution chart (based on current deque)")
                except Exception as e:
                    print(f"Error generating transformer layer {layer_idx} attention weight evolution chart: {e}")
                
        except Exception as e:
            print(f"Error generating analysis report: {e}")
    
    def export_history_to_csv(self):
        """Export historical data to CSV files.
        For all data types, only export snapshot of current data, don't clear data.
        """
        try:
            # Handle list-type data
            list_data = {}
            base_length = self.total_steps_recorded_for_list_history # Number of steps recorded in current interval
            
            if base_length == 0:
                print("No list-type data in current interval to export to training_history.csv")
                return
                
            # Prepare data for export
            for key in ['rewards', 'actions', 'epsilon', 'sf1_rewards', 'sf2_rewards', 'learning_rates', 'elapsed_times', 'losses']:
                if key in self.history and len(self.history[key]) > 0:
                    # Ensure exported data length matches base_length
                    if len(self.history[key]) >= base_length:
                        list_data[key] = self.history[key][-base_length:]  # Take the latest base_length items
                    else: # If list length is insufficient
                        list_data[key] = self.history[key] + [None] * (base_length - len(self.history[key]))
                elif key not in self.history: # If a key is not initialized
                    list_data[key] = [None] * base_length
            
            # Write to CSV, overwrite each time, only include data from current interval
            training_csv_path = self.output_dir / 'training_history.csv'
            pd.DataFrame(list_data).to_csv(training_csv_path, index=False)
            print(f"Current interval training history data exported to {training_csv_path}")
            
            # Handle deque-type states data
            try:
                if len(self.history['states']) > 0:
                    states_list = list(self.history['states']) # Convert deque to list
                    
                    # Convert to numpy array
                    states_np = np.array(states_list)
                    
                    # Ensure states_np is 2D (batch, features)
                    if states_np.ndim > 2: # For example (batch, seq_len, features) -> (batch*seq_len, features)
                        orig_shape = states_np.shape
                        states_np = states_np.reshape(states_np.shape[0], -1)
                        
                    # Convert to DataFrame for CSV export
                    columns = [f'state_{i}' for i in range(states_np.shape[1])]
                    states_df = pd.DataFrame(states_np, columns=columns)
                    
                    # Export to CSV
                    state_csv_path = self.output_dir / 'state_history_deque_snapshot.csv'
                    states_df.to_csv(state_csv_path, index=False)
                    print(f"States history (deque snapshot) exported to {state_csv_path}")
                else:
                    print("States deque is empty, not exporting state_history_deque_snapshot.csv")
            except Exception as e:
                print(f"Error exporting states history (deque): {e}")
        except Exception as e:
            print(f"Error exporting history to CSV: {e}")
            
    def save_transformer_attention_weights(self, step_idx=None, suffix=""):
        """Save Transformer layer attention weights heatmap, showing weights between time steps
        
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
            
            return  # Directly return after successfully getting weights from transformer layers
        
        # If unable to get directly, fall back to using history (for full visualization mode)
        if 'transformer_attn_weights' not in self.history or not isinstance(self.history['transformer_attn_weights'], list):
            print("Transformer attention weights history not initialized or format incorrect, and cannot be directly obtained from transformer layers")
            return

        for i, layer_history_deque in enumerate(self.history['transformer_attn_weights']):
            if not layer_history_deque: # If deque is empty
                print(f"Transformer layer {i} attention weights history (deque) is empty, cannot generate heatmap")
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
            print(f"No transformer layer {layer_idx} attention weights history data")
            return
            
        layer_history_deque = self.history['transformer_attn_weights'][layer_idx]
        if not layer_history_deque:
            print(f"Transformer layer {layer_idx} attention weights history (deque) is empty")
            return
            
        deque_len = len(layer_history_deque)
        if deque_len < num_snapshots:
            print(f"Not enough data in deque ({deque_len} snapshots) to display fewer than {num_snapshots} snapshots. Displaying all available snapshots.")
            indices_to_plot = range(deque_len)
            actual_num_snapshots = deque_len
        else:
            # Uniformly sample snapshot indices from the deque
            indices_to_plot = np.linspace(0, deque_len - 1, num_snapshots, dtype=int)
            actual_num_snapshots = num_snapshots

        if actual_num_snapshots == 0:
            print(f"Transformer layer {layer_idx} has no snapshots to plot.")
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
            ax.set_title(f"Snapshot {snapshot_idx_in_deque+1}/{deque_len}") # Display position in deque
            
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
    
    # Create test states
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
        print(f"Loaded old model: {old_model_path}")
        
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
        print(f"Loaded new model: {new_model_path}")
        
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