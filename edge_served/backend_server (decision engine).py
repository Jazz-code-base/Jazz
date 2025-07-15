import socket
import os
import struct
import torch
import numpy as np
import time
import signal
import sys
import threading
import pickle
from collections import deque
import math
import csv
import datetime
import matplotlib.pyplot as plt

# Import DRL model components
from dtqn_adapter import DTQN_adapter
from visualization import DTQNVisualizer

# Server configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 9000  # Listen port
REWARD_CSV_FILENAME = "reward_history.csv"  # Reward history filename

# State and action dimensions
SF_STATE_LENGTH = 8
SF_ACTION_NUM = 3
SF_NUM = 2
state_dim = SF_STATE_LENGTH * SF_NUM  # 8 features * 2 subflows
action_dim = pow(SF_ACTION_NUM, SF_NUM)  # 9 possible actions

# Other global variables
TOTAL_REWARD_MIN = -210
TOTAL_REWARD_MAX = 210
CWND_BOUND = (30, 2500)
DEMAND_TYPE = 0

# Training parameters
EPISODE_EQUALS_CONNECTION = False  # Set to True to make episode length equal to connection length
TRAIN_TIMESTEP_COUNT = 1 # Train every n steps
CON_DURATION = 500  # Connection reset cycle
EPISODE_LEN = 300 if not EPISODE_EQUALS_CONNECTION else CON_DURATION  # If alignment mode is enabled, use connection cycle as episode length
EXPLORE_INTERVAL = 6  # Exploration interval steps

# Set GPU device
print("============================================================================================")
# Check GPU availability
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")
print("============================================================================================")

# Model configuration
cfg = {
    'device': device,
    'gamma': 0.969,
    'epsilon_start':0.99,
    'epsilon_end':0.01,
    'epsilon_decay':150000,
    'lr':1e-4,
    'lr_decay': 0.96,        
    'lr_decay_train_counts': 10,  
    'lr_min': 1e-6,        
    'batch_size':32,
    'memory_capacity':1000000,    
    'hidden_dim':128,
    'n_states': state_dim,
    'n_actions': action_dim,
    'target_update_count': 10000,
    'model_save_interval': 50000,
   
    # DTQN specific parameters
    'history_len': 20,
    'context_len': 20,
    'bag_size': 0,
    'max_env_steps': EPISODE_LEN,
    'use_dual_flow': True,
   
    # Priority experience replay parameters
    'use_per': False,
    'per_alpha': 0.6,
    'per_beta_start': 0.1,
    'per_beta_frames': 80000,
    'per_eps': 1e-6,
   
    # DTQN network structure parameters
    'embed_per_obs_dim': 1,
    'action_embed_dim': 0,
    'inner_embed_size': 128,
    'num_heads': 4,
    'num_transformer_layers': 1,
    'dropout': 0.0,
    'gate': "res",
    'identity': False,
    'pos': "learned",
    'discrete': False,
}

# Model and training related
STATE_HISTORY_LEN = cfg['context_len']+5 # Keep recent state history for trend calculation
state_history = deque(maxlen=STATE_HISTORY_LEN)
start_time = None

# Action mapping table
action_mapping = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    4: (1, 1),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2)
}

# Visualization configuration
ENABLE_VISUALIZATION = False
ENABLE_CSV_REWARD_LOGGING = False
ENABLE_MINIMAL_HEATMAP_ONLY = False
VISUALIZATION_INTERVAL = 10000

# Global model instances
dtqn_adapter = None
visualizer = None
is_random_action = False
last_print_time = None  # Record the last print time

# Add global state variables, similar to test_server_explore.py
prev_state = None
prev_state_norm = None
prev_action = None

def clamp(value, min_value, max_value):
    """Limit value within specified range"""
    return max(min_value, min(value, max_value))

def normalize_state_for_printing(raw_state_vector):
    """Helper function to normalize a single state vector for printing."""
    if torch.is_tensor(raw_state_vector):
        state_to_normalize = raw_state_vector.cpu().numpy().copy()
    else:
        state_to_normalize = raw_state_vector.copy()

    normalized_vector = np.zeros_like(state_to_normalize, dtype=np.float32)

    # Define normalization ranges (consistent with process_state)
    tput_min, tput_max = 0, 2.5e-1
    rtt_min, rtt_max = 0, 30000
    effective_cwnd_min_for_norm = 0 
    effective_cwnd_max_for_norm = CWND_BOUND[1]
    loss_min, loss_max = 0, 0.01

    num_subflows = len(state_to_normalize) // SF_STATE_LENGTH

    for i in range(num_subflows):
        base_idx = i * SF_STATE_LENGTH

        # Extract features from original vector
        avg_tput = state_to_normalize[base_idx + 0]
        measured_bw = state_to_normalize[base_idx + 1]
        avg_rtt = state_to_normalize[base_idx + 2]
        min_rtt = state_to_normalize[base_idx + 3]
        cwnd = state_to_normalize[base_idx + 4]
        explore_flag = state_to_normalize[base_idx + 5]
        loss_rate = state_to_normalize[base_idx + 6]
        demand_type_val = state_to_normalize[base_idx + 7]

        # Perform min-max normalization
        avg_tput_norm = (avg_tput - tput_min) / (tput_max - tput_min) if (tput_max - tput_min) != 0 else 0
        measured_bw_norm = (measured_bw - tput_min) / (tput_max - tput_min) if (tput_max - tput_min) != 0 else 0
        avg_rtt_norm = (avg_rtt - rtt_min) / (rtt_max - rtt_min) if (rtt_max - rtt_min) != 0 else 0
        min_rtt_norm = (min_rtt - rtt_min) / (rtt_max - rtt_min) if (rtt_max - rtt_min) != 0 else 0
        cwnd_norm = (cwnd - effective_cwnd_min_for_norm) / (effective_cwnd_max_for_norm - effective_cwnd_min_for_norm) if (effective_cwnd_max_for_norm - effective_cwnd_min_for_norm) != 0 else 0
        loss_rate_norm = (loss_rate - loss_min) / (loss_max - loss_min) if (loss_max - loss_min) != 0 else 0
       
        # Normalization for explore_flag and demand_type is consistent with subflow_state_norm in process_state
        explore_flag_norm = explore_flag / 2.0
        demand_type_norm = demand_type_val / 2.0

        # Limit normalized values to [0,1] range
        norm_values = [
            avg_tput_norm, measured_bw_norm, avg_rtt_norm, min_rtt_norm,
            cwnd_norm, explore_flag_norm, loss_rate_norm, demand_type_norm
        ]
        for j, val_norm in enumerate(norm_values):
            normalized_vector[base_idx + j] = clamp(val_norm, 0, 1)
           
    return normalized_vector

def calculate_threshold(min_rtt, base_product=9500, growth_factor=0.003, min_rtt_range=(700, 20000)):
    """
    Parameters of this function can be adjusted to adapt to optimal BDP points in different network environments or to meet specific requirements (e.g., low latency), with future work focusing on more accurate delay modeling.

    Calculate dynamic threshold such that: thresh × min_rtt = base_product × (1 + growth_factor × (min_rtt - min_rtt_range[0])/1000)
    """
    min_rtt = np.clip(min_rtt, *min_rtt_range)
    normalized_rtt = (min_rtt - min_rtt_range[0]) / 100  # Normalize per 1000ms
    dynamic_product = base_product * (1 + growth_factor * normalized_rtt)
    return dynamic_product / min_rtt

def process_state(state_dic, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step):
    """Process state and perform normalization"""
    state = []
    state_norm = []

    # Define normalization ranges
    tput_min, tput_max = 0, 2.5e-1  # Throughput normalization range
    rtt_min, rtt_max = 0, 30000     # RTT normalization range
    cwnd_min, cwnd_max = 0, CWND_BOUND[1]
    loss_min, loss_max = 0, 0.01

    for ip in sorted(state_dic.keys()):
        avg_tput = tput_history[ip][0] / tput_history[ip][1]
        avg_rtt = sum(rtt_history[ip]) / len(rtt_history[ip])
        if avg_rtt > 1000000:
            avg_rtt = 6000
        loss_rate = 0  # Not considering packet loss for now

        measured_bw = state_dic[ip][5] / (2 ** 24)
        min_rtt = state_dic[ip][4]
        cwnd = state_dic[ip][6]

        # Perform min-max normalization
        avg_tput_norm = (avg_tput - tput_min) / (tput_max - tput_min)
        measured_bw_norm = (measured_bw - tput_min) / (tput_max - tput_min)
        avg_rtt_norm = (avg_rtt - rtt_min) / (rtt_max - rtt_min)
        min_rtt_norm = (min_rtt - rtt_min) / (rtt_max - rtt_min)
        cwnd_norm = (cwnd - cwnd_min) / (cwnd_max - cwnd_min)
        loss_rate_norm = (loss_rate - loss_min) / (loss_max - loss_min)

        # Limit normalized values to [0,1] range
        avg_tput_norm = clamp(avg_tput_norm, 0, 1)
        measured_bw_norm = clamp(measured_bw_norm, 0, 1)
        avg_rtt_norm = clamp(avg_rtt_norm, 0, 1)
        min_rtt_norm = clamp(min_rtt_norm, 0, 1)
        cwnd_norm = clamp(cwnd_norm, 0, 1)
        loss_rate_norm = clamp(loss_rate_norm, 0, 1)

        time_since_last_inc = time_step - last_inc_steps.get(ip, EXPLORE_INTERVAL*10)

        if (force_dec_flag[ip] == 666):
            explore_flag = 2.0
        elif (time_since_last_inc > EXPLORE_INTERVAL) and (cwnd < (CWND_BOUND[1]-25)):      
            explore_flag = 1.0  # Need to perform INC
        else:
            explore_flag = 0.0
        
        explore_flag_norm = explore_flag / 2

        subflow_state = [avg_tput, measured_bw, avg_rtt, min_rtt, cwnd, explore_flag, loss_rate, DEMAND_TYPE / 2]
        subflow_state_norm = [avg_tput_norm, measured_bw_norm, avg_rtt_norm, min_rtt_norm, cwnd_norm, explore_flag_norm, loss_rate_norm, DEMAND_TYPE / 2]
        state.extend(subflow_state)
        state_norm.extend(subflow_state_norm)
       
    state = torch.tensor(state, dtype=torch.float32, device=dtqn_adapter.device)
    state_norm = torch.tensor(state_norm, dtype=torch.float32, device=dtqn_adapter.device)
    return state, state_norm 

class BackendServer:
    def __init__(self, host, port):
        """Initialize backend server"""
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = []
        self.lock = threading.Lock()
        
    def start(self):
        """Start server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            print(f"Backend server started, listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_sock, client_addr = self.socket.accept()
                    print(f"Accepted connection from {client_addr}")
                    
                    # Create a thread for each client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_sock, client_addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    with self.lock:
                        self.clients.append((client_sock, client_addr, client_thread))
                except Exception as e:
                    if self.running:  # Only report errors during normal operation
                        print(f"Error accepting client connection: {e}")
                        
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            self.cleanup()
            
    def handle_client(self, client_sock, client_addr):
        """Handle client connection and requests"""
        try:
            while self.running:
                # Receive data length
                size_data = client_sock.recv(4)
                if not size_data:
                    print(f"Client {client_addr} disconnected")
                    # Client disconnected, automatically reset state
                    self.handle_client_disconnect()
                    break
                
                data_size = struct.unpack('!I', size_data)[0]
                
                # Receive data
                chunks = []
                bytes_recvd = 0
                while bytes_recvd < data_size:
                    chunk = client_sock.recv(min(data_size - bytes_recvd, 4096))
                    if not chunk:
                        print(f"Client {client_addr} connection interrupted during data reception")
                        self.handle_client_disconnect()
                        raise ConnectionError("Data reception interrupted")
                    chunks.append(chunk)
                    bytes_recvd += len(chunk)
                
                # Deserialize data
                data = pickle.loads(b''.join(chunks))
                
                # Process data
                response = self.process_data(data)
                
                # Send response
                serialized_response = pickle.dumps(response)
                response_size = len(serialized_response)
                client_sock.sendall(struct.pack('!I', response_size))
                client_sock.sendall(serialized_response)
                
        except ConnectionError:
            print(f"Client {client_addr} connection interrupted")
            self.handle_client_disconnect()
        except Exception as e:
            print(f"Error processing client {client_addr} data: {e}")
            self.handle_client_disconnect()
        finally:
            try:
                client_sock.close()
            except:
                pass
            
            # Remove from client list
            with self.lock:
                self.clients = [(s, a, t) for s, a, t in self.clients if a != client_addr]
    
    def handle_client_disconnect(self):
        """Handle client disconnection"""
        global prev_state, prev_state_norm, prev_action, state_history, dtqn_adapter
        
        print("Handling client disconnect - resetting backend state...")
        
        # If there's an unfinished episode, end it first
        if prev_state_norm is not None and prev_action is not None and dtqn_adapter is not None:
            try:
                # Create a termination transition
                dummy_next_state = torch.zeros_like(prev_state_norm)
                dummy_reward = torch.tensor(0.0, device=dtqn_adapter.device)
                done_tensor = torch.tensor(1.0, device=dtqn_adapter.device)
                
                # Add termination transition to replay buffer
                dtqn_adapter.observe(prev_state_norm, prev_action, reward=dummy_reward, next_state=dummy_next_state, done=done_tensor)
                print("Added termination transition for disconnected client")
                
                # End current episode
                dtqn_adapter.dtqn_agent.replay_buffer.flush()
                print("Episode flushed due to client disconnect")
                
            except Exception as e:
                print(f"Error handling termination transition: {e}")
        
        # Reset state variables
        prev_state = None
        prev_state_norm = None
        prev_action = None
        state_history.clear()
        
        print("Backend state reset complete - ready for new connection")
    
    def process_data(self, data):
        """Process data sent by client and generate response"""
        global state_history, dtqn_adapter, visualizer, prev_action
        
        try:
            data_type = data.get("data_type")
            
            # If reset command
            if data_type == "reset":
                return self.handle_reset(data)
            
            # Normal state processing
            state_dict = data.get("state_dict", {})
            tput_history = data.get("tput_history", {})
            rtt_history = data.get("rtt_history", {})
            prev_action = data.get("prev_action")
            time_step = data.get("time_step", 0)
            last_inc_steps = data.get("last_inc_steps", {})
            force_dec_flag = data.get("force_dec_flag", {})
            con_init_flag = data.get("con_init_flag", 0)
            
            # Process state
            next_state, next_state_norm, sf1_reward, sf2_reward, total_reward = self.process_state_and_compute_reward(
                state_dict, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step, con_init_flag, data
            )
            
            # If not initial state
            if not (data_type == 0 and con_init_flag):
                # Update DQN model
                self.update_dqn_and_memory(total_reward, next_state_norm, time_step, sf1_reward, sf2_reward)
            
            # Prepare next action
            action_tuple = self.prepare_next_action(next_state, next_state_norm, data_type)
            
            # Prepare response - ensure all tensors are moved to CPU
            def tensor_to_cpu(obj):
                """Recursively move tensors to CPU"""
                if torch.is_tensor(obj):
                    return obj.cpu()
                elif isinstance(obj, dict):
                    return {k: tensor_to_cpu(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(tensor_to_cpu(item) for item in obj)
                else:
                    return obj
            
            response = {
                "next_state": tensor_to_cpu(next_state),
                "action": tensor_to_cpu(prev_action) if prev_action is not None else None,
                "action_tuple": action_tuple,
                "force_dec_flag": force_dec_flag,
                "last_inc_steps": last_inc_steps,
                "initial_phase_end": 1 if data_type == 0 else 0,
                "feedback_phase_end": 1 if data_type == 1 else 0,
                "con_init_flag": 0 if data_type == 0 and con_init_flag else con_init_flag
            }
            
            return response
            
        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def handle_reset(self, data):
        """Handle connection reset request"""
        global dtqn_adapter, prev_state, prev_state_norm, prev_action, state_history
        
        print("Handling connection reset request...")
        
        received_prev_state = data.get("prev_state")
        received_prev_action = data.get("prev_action")
        
        # If there was a previous state and the DTQN adapter exists, handle the last transition
        if received_prev_state is not None and received_prev_action is not None and dtqn_adapter is not None:
            try:
                # Create a termination transition for the last state
                dummy_next_state = torch.zeros_like(torch.tensor(received_prev_state, device=dtqn_adapter.device))
                dummy_reward = torch.tensor(0.0, device=dtqn_adapter.device)
                done_tensor = torch.tensor(1.0, device=dtqn_adapter.device)
                
                # If previous state exists, mark it as terminal
                if prev_state_norm is not None and prev_action is not None:
                    dtqn_adapter.observe(prev_state_norm, prev_action, reward=dummy_reward, next_state=dummy_next_state, done=done_tensor)
                    print("Added termination transition to replay buffer")
                
                # Explicitly call flush to end current episode
                dtqn_adapter.dtqn_agent.replay_buffer.flush()
                print("Current episode flushed in replay buffer")
                
            except Exception as e:
                print(f"Error handling last transition during reset: {e}")
        
        # Reset state variables (but keep training-related caches)
        prev_state = None
        prev_state_norm = None  
        prev_action = None
        state_history.clear()
        
        print("Backend state variables reset successfully")
        print("Training cache and model preserved")
        
        return {"status": "reset_handled"}
    
    def process_state_and_compute_reward(self, data_dict, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step, con_init_flag, data):
        """Process state and compute reward"""
        global state_history, prev_state
        
        next_state, next_state_norm = process_state(data_dict, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step)
        
        if not (data_type == 0 and con_init_flag):
            # Get current history state list (excluding next_state)
            current_history = list(state_history)
            # Get prev_action from client data
            client_prev_action = data.get("prev_action") if data else None
            sf1_reward, sf2_reward, total_reward = self.compute_reward(next_state, prev_state, client_prev_action, current_history, time_step)
            
            # After computing reward, add prev_state to history
            if prev_state is not None:
                state_history.append(prev_state.cpu().numpy())
                
            return next_state, next_state_norm, sf1_reward, sf2_reward, total_reward
        else:
            # Also record for initial state
            if next_state is not None:
                state_history.append(next_state.cpu().numpy())
            return next_state, next_state_norm, None, None, None
    
    def stop(self):
        """Stop server"""
        self.running = False
        
        # Close all client connections
        with self.lock:
            for client_sock, _, _ in self.clients:
                try:
                    client_sock.close()
                except:
                    pass
            self.clients = []
        
        # Close server socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
                
        print("Backend server stopped")
        
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        
    def compute_reward(self, next_state, prev_state, prev_action, history_states=None, time_step=None):
        """Calculate reward"""
        SF_WEIGHT = (1, 1)
        CONTEXT_WINDOW = cfg['context_len']+1  # History window size for trend calculation

        # Convert tensor to numpy array to avoid GPU tensor access issues
        if torch.is_tensor(next_state):
            next_state_np = next_state.cpu().numpy()
        else:
            next_state_np = next_state
            
        if torch.is_tensor(prev_state):
            prev_state_np = prev_state.cpu().numpy()
        else:
            prev_state_np = prev_state

        def calculate_trend(data_series):
            """Calculate linear trend (slope)"""
            if len(data_series) < 2:
                return 0.0
            time_axis = np.arange(len(data_series))
            # Check if there is enough variation to calculate slope
            if len(np.unique(data_series)) < 2:
                 return 0.0 # If all values are the same, slope is 0
            try:
                slope = np.polyfit(time_axis, data_series, 1)[0]
                return slope
            except (np.linalg.LinAlgError, ValueError):
                 # Handle cases where fitting fails
                 return 0.0

        # --- Main function logic ---
        reward_list = []
        if prev_state is None: # Handle initial state case
            return 0.0, 0.0, 0.0 # Or return other default values

        sf_state_len = len(next_state_np) / 2
        temp_tuple = action_mapping[prev_action.item()] if torch.is_tensor(prev_action) else action_mapping[prev_action]
       
        # Check if any subflow in previous state hit boundary
        prev_any_sf_hit_boundary = False
        for i in range(2):
            base_idx = int(i * sf_state_len)
            prev_cwnd = prev_state_np[4 + base_idx]
            if prev_cwnd <= CWND_BOUND[0] or prev_cwnd >= CWND_BOUND[1]:
                prev_any_sf_hit_boundary = True
                break
        
        print(f"prev_any_sf_hit_boundary:{prev_any_sf_hit_boundary}")
       
        # Check if any subflow in current state hit boundary
        any_sf_hit_boundary = False
        for i in range(2):
            base_idx = int(i * sf_state_len)
            curr_cwnd = next_state_np[4 + base_idx]
            if curr_cwnd <= CWND_BOUND[0] or curr_cwnd >= CWND_BOUND[1]:
                any_sf_hit_boundary = True
                break
       
        def calculate_sf_reward(subflow_index, history_states=None):
            base_idx = int(subflow_index * sf_state_len)

            # --- Start: Subflow reward calculation preparation phase -------------------------------------------------------------------------------------------------------------------------
            # --- Get current and previous states from next_state and prev_state ---
            curr_tput = next_state_np[0 + base_idx]
            prev_tput = prev_state_np[0 + base_idx]
            curr_rtt = next_state_np[2 + base_idx]
            prev_rtt = prev_state_np[2 + base_idx]
            curr_min_rtt = next_state_np[3 + base_idx]
            prev_min_rtt = prev_state_np[3 + base_idx]
            curr_cwnd = next_state_np[4 + base_idx]
            prev_cwnd = prev_state_np[4 + base_idx]
            curr_measured_bw = next_state_np[1 + base_idx]
            prev_measured_bw = prev_state_np[1 + base_idx]
            curr_loss_rate = next_state_np[6 + base_idx]  # Assuming loss rate still comes from prev_state
            prev_loss_rate = prev_state_np[6 + base_idx]  # Assuming loss rate still comes from prev_state
            explore_flag = prev_state_np[5 + base_idx]
            action = temp_tuple[subflow_index]

            # --- Calculate historical trend (if history is long enough) ---
            rtt_trend = 0.0
            tput_trend = 0.0

            # Build sequence for trend calculation (oldest -> ... -> prev_state -> next_state)
            rtt_values_in_window = []
            tput_values_in_window = []

            # 1. Get states from history_states
            if CONTEXT_WINDOW >= 3: # Need at least one state from history_states
                num_needed_from_history = CONTEXT_WINDOW - 2
                if history_states and len(history_states) >= num_needed_from_history:
                    rtt_values_in_window.extend([s[2 + base_idx] for s in history_states[-num_needed_from_history:]])
                    tput_values_in_window.extend([s[0 + base_idx] for s in history_states[-num_needed_from_history:]])

            # 2. Add prev_state's rtt and tput (if TREND_WINDOW >= 2)
            if CONTEXT_WINDOW >= 2:
                # prev_rtt and prev_tput are already extracted from prev_state
                rtt_values_in_window.append(prev_rtt)
                tput_values_in_window.append(prev_tput)
           
            # 3. Add next_state's rtt and tput (if TREND_WINDOW >= 1)
            if CONTEXT_WINDOW >= 1:
                # curr_rtt and curr_tput are already extracted from next_state
                rtt_values_in_window.append(curr_rtt)
                tput_values_in_window.append(curr_tput)
           
            # Calculate trend (requires at least 2 points)
            if len(rtt_values_in_window) >= 2:
                rtt_trend = calculate_trend(rtt_values_in_window)
                # print(f"Subflow {subflow_index}: RTT Trend Seq: {[f'{x:.2f}' for x in rtt_values_for_trend]}, Trend: {rtt_trend:.2f}")
            # else:
                # print(f"Subflow {subflow_index}: Not enough data for RTT trend (need >= 2, got {len(rtt_values_for_trend)})")

            if len(tput_values_in_window) >= 2:
                tput_trend = calculate_trend(tput_values_in_window)
                # print(f"Subflow {subflow_index}: Tput Trend Seq: {[f'{x:.4f}' for x in tput_values_for_trend]}, Trend: {tput_trend:.4f}")
            # else:
                # print(f"Subflow {subflow_index}: Not enough data for Tput trend (need >= 2, got {len(tput_values_for_trend)})")

            # Calculate weighted average RTT and throughput in the window (newer samples have higher weight)
            avg_rtt_in_window = 0.0
            avg_tput_in_window = 0.0

            if rtt_values_in_window: # Ensure list is not empty
                # avg_rtt_in_window = np.mean(rtt_values_in_window) # Original average
                # Create linearly increasing weights: [1, 2, 3, ..., n], newest sample has highest weight
                weights = np.arange(1, len(rtt_values_in_window) + 1, dtype=np.float32)
                weights = weights / np.sum(weights)  # Normalize weights
                avg_rtt_in_window = np.average(rtt_values_in_window, weights=weights)
                # print(f"Subflow {subflow_index}: Weighted Avg RTT in window: {avg_rtt_in_window:.2f}")
           
            if tput_values_in_window: # Ensure list is not empty
                # avg_tput_in_window = np.mean(tput_values_in_window) # Original average
                # Create linearly increasing weights: [1, 2, 3, ..., n], newest sample has highest weight
                weights = np.arange(1, len(tput_values_in_window) + 1, dtype=np.float32)
                weights = weights / np.sum(weights)  # Normalize weights
                avg_tput_in_window = np.average(tput_values_in_window, weights=weights)
                # print(f"Subflow {subflow_index}: Weighted Avg Tput in window: {avg_tput_in_window:.4f}")

            # --- End: Subflow reward calculation preparation phase -----------------------------------------------------------------------------------------------------------------------------
            # --- Reward calculation logic ---
            rtt_thresh_weight = 0.5
            sf_strong_punish = 0
            sf_reward = 0

            # Modified boundary penalty logic
            if any_sf_hit_boundary:
                sf_strong_punish = -99
                # if curr_cwnd <= CWND_BOUND[0]:
                #     sf_strong_punish = -99 # If any subflow hits boundary, all subflows are punished
                # else:
                #     sf_strong_punish = -39
                sf_reward = 0
            elif prev_any_sf_hit_boundary and not any_sf_hit_boundary:
                # If previous state hit boundary but current state is normal, give all subflows a 99 reward
                sf_strong_punish = 99
                sf_reward = 0
            elif (explore_flag == 1 and action != 1) or (explore_flag == 2 and action != 2):
                # If explore_flag=1 but this subflow's actual action is not INC, reward is -99
                sf_strong_punish = -99
                sf_reward = 0
            else:
                # BDP = curr_measured_bw * curr_min_rtt if curr_min_rtt > 0 else 0

                # # 1. BDP related reward (based on prev_state)
                # if (curr_cwnd < 2 * BDP) and (curr_loss_rate == 0):
                #      # Keep original logic, encourage increasing window when not reaching BDP
                #     #  sf_reward = (prev_cwnd - 2 * BDP) + (curr_cwnd - prev_cwnd) / 4
                #      sf_reward = (curr_cwnd - 2 * BDP)
                # else:
                    # 2. RTT and throughput trade-off (combine current state and historical trend)
                    RTT_WARNING_THRESH = calculate_threshold(curr_min_rtt) # Use prev_min_rtt to calculate threshold
                    k = 6 # Assuming no packet loss

                    rtt_ratio = avg_rtt_in_window / curr_min_rtt if curr_min_rtt > 0 else 1.0
                    alpha = 1 / (1 + math.exp(-k * (rtt_ratio - RTT_WARNING_THRESH)))

                    # RTT penalty: Combine state threshold and historical trend
                    # Penalize states exceeding threshold + penalize increasing RTT trend
                    # Need to adjust coefficients to balance both parts
                    rtt_state_penalty = -rtt_thresh_weight * (rtt_ratio - RTT_WARNING_THRESH) / 0.05
                    rtt_trend_penalty = -0 * rtt_trend # Penalize positive slope (increasing RTT), coefficient is empirical, needs adjustment
                    rtt_penalty = rtt_state_penalty + rtt_trend_penalty
                    # rtt_penalty = rtt_state_penalty
                    # print(f"rtt_state_penalty:{rtt_state_penalty}, rtt_trend_penalty:{rtt_trend_penalty}, rtt_penalty:{rtt_penalty}")
                    # print(f"rtt_penalty:{rtt_penalty}")

                    # Throughput reward: Combine absolute throughput and historical trend
                    # Reward high throughput states + reward increasing throughput trend
                    # Need to adjust coefficients
                    tput_state_reward = avg_tput_in_window / 0.001 # Based on previous throughput
                    tput_trend_reward = 0 * tput_trend # Reward positive slope (increasing throughput), coefficient 60 is empirical
                    tput_reward = tput_state_reward + tput_trend_reward
                    # tput_reward = tput_state_reward
                    # print(f"tput_state_reward:{tput_state_reward}, tput_trend_reward:{tput_trend_reward}, tput_reward:{tput_reward}")
                    # print(f"tput_reward:{tput_reward}")

                    # Dynamic weighted combination
                    sf_reward = alpha * rtt_penalty + (1 - alpha) * tput_reward

            # Combine final reward
            final_sf_reward = SF_WEIGHT[subflow_index] * sf_reward + sf_strong_punish
            return final_sf_reward

        # --- Calculate reward and return -----------------------------------------------------------------------------------------------------------------------------
        for i in range(2):
            sf_reward_val = calculate_sf_reward(i, history_states)
            reward_list.append(sf_reward_val)

        total_reward = sum(reward_list)

        # Ensure two subflow rewards and total reward are returned
        # sf1_r = reward_list[0] if len(reward_list) > 0 else 0.0
        # sf2_r = reward_list[1] if len(reward_list) > 1 else 0.0
        sf1_r = reward_list[0]
        sf2_r = reward_list[1]
        
        # --- Add state printing functionality (similar to test_server_explore.py) ---
        try:
            # Simplified printing functionality, only print current state information
            if next_state_np is not None:
                global last_print_time
                current_time = time.time()
                
                # Calculate time interval
                time_interval = ""
                if last_print_time is not None:
                    interval_ms = (current_time - last_print_time) * 1000
                    time_interval = f" (interval: {interval_ms:.1f}ms)"
                last_print_time = current_time
                
                print("=" * 80)
                print(f"Time Step: {time_step if time_step is not None else 'N/A'}{time_interval}")
                
                # Get action information
                action_info = ""
                if prev_action is not None:
                    if torch.is_tensor(prev_action):
                        action_idx = prev_action.item()
                    else:
                        action_idx = prev_action
                    action_tuple = action_mapping.get(action_idx, (0, 0))
                    action_names = ["DEC", "KEEP", "INC"]
                    action_info = f" | Actions: SF1={action_names[action_tuple[0]]}, SF2={action_names[action_tuple[1]]}"
                
                # Print current state information for each subflow
                for i in range(2):
                    source_ip = f"192.168.213.{i+2}"  # Simulate source IP
                    base_idx = i * int(len(next_state_np) / 2)
                    
                    # Set color codes
                    color_code = "\033[31m" if i == 0 else "\033[32m"  # Red or Green
                    color_reset = "\033[0m"
                    
                    # Only print current state information
                    print(f"{color_code}[SF{i+1}] source_ip:{source_ip}, cwnd:{next_state_np[4+base_idx]:.0f}, tput:{next_state_np[0+base_idx] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, max_tput:{next_state_np[1+base_idx] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, rtt:{int(next_state_np[2+base_idx])}, minrtt:{next_state_np[3+base_idx]:.0f}, reward:{reward_list[i]:.2f}{color_reset}")
                    
                print(f"Total Reward: {total_reward:.2f}{action_info}")
                print("=" * 80)
        except Exception as e:
            print(f"Error printing state information: {e}")
            
        return sf1_r, sf2_r, total_reward
        
    def update_dqn_and_memory(self, total_reward, next_state_norm, time_step, sf1_reward=None, sf2_reward=None):
        """Update DQN and experience replay memory"""
        global start_time, dtqn_adapter, visualizer, prev_state, prev_state_norm, prev_action
       
        # Determine if the current step marks the end of an episode
        if EPISODE_EQUALS_CONNECTION:
            # When alignment mode is enabled, only CON_DURATION is used as the episode end flag
            current_step_done_float = 1.0 if (time_step % CON_DURATION == CON_DURATION - 1) else 0.0
        else:
            # When alignment mode is not enabled, use the original condition: EPISODE_LEN or CON_DURATION satisfied to end
            current_step_done_float = 1.0 if (time_step % EPISODE_LEN == EPISODE_LEN - 1) or (time_step % CON_DURATION == CON_DURATION - 1) else 0.0
       
        current_step_done_tensor = torch.as_tensor(current_step_done_float, device=dtqn_adapter.device)
       
        total_reward = clamp(total_reward, -600, 600)
        total_reward_norm = (total_reward - TOTAL_REWARD_MIN) / (TOTAL_REWARD_MAX - TOTAL_REWARD_MIN) * 2 - 1
        total_reward_norm = torch.as_tensor(clamp(total_reward_norm, -1.0, 1.0), dtype=torch.float32, device=dtqn_adapter.device)
        print(f"norm reward:{total_reward_norm}")
        # Use observe method to add transition to DTQN agent's context
        if prev_action is not None:
            dtqn_adapter.observe(prev_state_norm, prev_action, reward=total_reward_norm, next_state=next_state_norm, done=current_step_done_tensor)
       
        # If the current step marks the end of an episode
        if current_step_done_float > 0.0:
            print(f"====== Episode Complete ======")
            print(f"Current step: {time_step}")
            print(f"Episode length: {EPISODE_LEN if not EPISODE_EQUALS_CONNECTION else CON_DURATION}")
            print(f"=========================")
            dtqn_adapter.dtqn_agent.replay_buffer.flush() # Explicitly call flush to end current buffer episode
            prev_state = None # Prepare for context_reset of next episode

        # ---- CSV logging logic ----
        if ENABLE_CSV_REWARD_LOGGING and prev_action is not None: # Add switch check
            try:
                action_to_log = prev_action.item() if torch.is_tensor(prev_action) else prev_action
                sf1_to_log = sf1_reward.item() if torch.is_tensor(sf1_reward) and sf1_reward is not None else sf1_reward
                sf2_to_log = sf2_reward.item() if torch.is_tensor(sf2_reward) and sf2_reward is not None else sf2_reward
                current_lr = dtqn_adapter.optimizer.param_groups[0]['lr'] if hasattr(dtqn_adapter, 'optimizer') and dtqn_adapter.optimizer.param_groups else 'N/A'
                elapsed_t = (time.time() - start_time) * 1000 if start_time is not None and time_step % 10 == 0 else None

                # Get beta value
                current_beta = 'N/A'
                if dtqn_adapter.use_per and hasattr(dtqn_adapter.dtqn_agent.replay_buffer, 'get_beta'):
                    current_beta = f"{dtqn_adapter.dtqn_agent.replay_buffer.get_beta():.4f}"

                row_data = [
                    time_step,
                    sf1_to_log,
                    sf2_to_log,
                    total_reward, # Record original total_reward
                    current_lr,
                    dtqn_adapter.epsilon,
                    action_to_log,
                    current_beta, # Beta value first
                    f"{elapsed_t:.2f}" if elapsed_t is not None else "" # Then Elapsed Time
                ]
                with open(REWARD_CSV_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_data)
            except Exception as e:
                print(f"Error writing reward to CSV {REWARD_CSV_FILENAME}: {e}")
       
        # Display buffer status
        buffer = dtqn_adapter.dtqn_agent.replay_buffer
        print(f"Buffer status - pos: {buffer.pos}, episodes: {buffer.pos[0]}")
       
        # If visualization is enabled, record step data
        if ENABLE_VISUALIZATION:
            if not ENABLE_MINIMAL_HEATMAP_ONLY: # Only record full data in non-minimal mode
                elapsed_time = time.time() - start_time if time_step % 10 == 0 else None
                if prev_action is not None:
                    visualizer.record_step(
                        state=prev_state_norm, 
                        action=prev_action.item(), 
                        reward=total_reward_norm.item(), 
                        epsilon=dtqn_adapter.epsilon,
                        sf1_reward=sf1_reward.item() if torch.is_tensor(sf1_reward) else sf1_reward,
                        sf2_reward=sf2_reward.item() if torch.is_tensor(sf2_reward) else sf2_reward,
                        elapsed_time=elapsed_time
                    )
       
        # Update model every TRAIN_TIMESTEP_COUNT steps
        if ((time_step != 0) and (time_step % TRAIN_TIMESTEP_COUNT == 0)):
            # Directly try to update model, let adapter handle check logic
            print(f"Start updating model, timestep: {time_step}, batch_size: {dtqn_adapter.batch_size}")
            start = time.perf_counter()
            train_success = dtqn_adapter.update()
            update_time = (time.perf_counter() - start) * 1000
           
            # if train_success:
            #     print(f"Model updated: {update_time:.2f} ms, training successful! Current total training count: {dtqn_adapter.train_count}")
            #     print(f"buffer sample count: {dtqn_adapter.dtqn_agent.replay_buffer.pos[0]}")
            # else:
            #     print(f"Model updated: {update_time:.2f} ms, training skipped - buffer sample insufficient")
            #     print(f"buffer sample count: {dtqn_adapter.dtqn_agent.replay_buffer.pos[0]}")
       
        # Periodically generate visualizations
        if (time_step != 0):
            if ENABLE_MINIMAL_HEATMAP_ONLY and (time_step % VISUALIZATION_INTERVAL == 0):
                try:
                    visualizer.save_attention_weights(step_idx=time_step)
                    print(f"Generated Transformer Heatmap at step {time_step}")
                except Exception as e:
                    print(f"Error generating Transformer Heatmap: {e}")
            elif ENABLE_VISUALIZATION and (time_step % VISUALIZATION_INTERVAL == 0):
                try:
                    visualizer.plot_reward_history(window_size=1000)
                    visualizer.plot_action_distribution()
                    visualizer.plot_epsilon_decay()
                   
                    # Visualize attention weights and Bag contents
                    visualizer.save_attention_weights(step_idx=time_step)
                   
                    # If bag mechanism is enabled, visualize bag contents
                    if hasattr(dtqn_adapter.dtqn_agent, 'bag') and dtqn_adapter.dtqn_agent.bag_size > 0:
                        visualizer.visualize_bag_contents(step_idx=time_step)
                   
                    # Export history data
                    visualizer.export_history_to_csv()
                   
                    print(f"Generated visualization results at step {time_step}")
                except Exception as e:
                    print(f"Error generating visualizations: {e}")
    
    def prepare_next_action(self, next_state, next_state_norm, data_type):
        """Prepare next action"""
        global dtqn_adapter, is_random_action, prev_state, prev_state_norm, prev_action
       
        prev_state, prev_state_norm = next_state, next_state_norm
       
        action, is_random = dtqn_adapter.sample_action(prev_state_norm)
        action_tuple = action_mapping[action]
        is_random_action = is_random  # Save random action flag
        prev_action = torch.tensor(action, device=dtqn_adapter.device, dtype=torch.long)
       
        return action_tuple

    # Other methods will be added later... 

def init_model():
    """Initialize model and visualizer"""
    global dtqn_adapter, visualizer, start_time, cfg, last_print_time
    
    # Initialize time variables
    last_print_time = None
    
    # Initialize model
    model_path = "saved_models/dtqn_model_latest_state.pth"  # Specify path for pre-trained model if available
    if os.path.exists(model_path):
        try:
            dtqn_adapter = DTQN_adapter(cfg)
            dtqn_adapter.load_model(model_path)
            print(f"Pre-trained model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}, initializing new model.")
            dtqn_adapter = DTQN_adapter(cfg)
    else:
        dtqn_adapter = DTQN_adapter(cfg)
    
    # Add visualization tool
    visualizer = DTQNVisualizer(dtqn_adapter)
    
    # Model warm-up
    dummy_state = torch.FloatTensor(np.random.rand(state_dim)).to(dtqn_adapter.device)
    with torch.no_grad():
        _ = dtqn_adapter.sample_action(dummy_state)
    print("Model warm-up completed")
    
    # Create or clear reward file
    if ENABLE_CSV_REWARD_LOGGING:
        clear_reward_file()
    
    # Set start time
    start_time = time.time()
    
    return dtqn_adapter, visualizer

def clear_reward_file(filename=REWARD_CSV_FILENAME):
    """Clear reward file and write header"""
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time Step", "sf1_reward", "sf2_reward", "total_reward", "Learning Rate", "Epsilon", "Action", "Beta", "Elapsed Time (ms)"])
        print("Reward file cleared and initialized")
    except Exception as e:
        print(f"Failed to clear reward file: {e}")

def signal_handler(sig=None, frame=None):
    """Signal handler function"""
    if hasattr(signal_handler, '_called'):
        return
    signal_handler._called = True
    
    print('Forced exit!')
    
    try:
        # Generate analysis report before exit
        if ENABLE_VISUALIZATION and not ENABLE_MINIMAL_HEATMAP_ONLY:
            visualizer.generate_analysis_report()
            visualizer.plot_reward_history()
            visualizer.plot_action_distribution()
            visualizer.export_history_to_csv()
            print("Final analysis report and visualizations generated")
    except Exception as e:
        print(f"Error generating analysis report: {e}")
    
    try:
        # Save final model
        if dtqn_adapter is not None:
            model_path = f"dtqn_model_final_{dtqn_adapter.train_count}.pth"
            # dtqn_adapter.save_model(model_path)
            # print(f"Final model saved: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    sys.exit(0)

def main():
    """Main function"""
    global dtqn_adapter, visualizer
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize model
    dtqn_adapter, visualizer = init_model()
    
    # Create server instance
    server = BackendServer(HOST, PORT)
    
    try:
        # Start server
        server.start()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, server will shut down")
    except Exception as e:
        print(f"Error running server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Try to gracefully shut down the server
        try:
            server.stop()
        except:
            pass
        
        # Call signal handler to clean up resources
        signal_handler()

if __name__ == "__main__":
    main() 