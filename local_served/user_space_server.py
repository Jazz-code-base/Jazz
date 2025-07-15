import socket
import os
import selectors
import struct
import errno
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import math
from dtqn_adapter import DTQN_adapter
from visualization import DTQNVisualizer, compare_models
import subprocess
import signal
import sys
import atexit
from collections import deque
DEMAND_TYPE = 0

SOCKET_PATH = "/tmp/mysocket"
BUFFER_SIZE = 1024
REWARD_CSV_FILENAME = "reward_history.csv" # Global CSV filename

# Define server structure format for receiving u32 data
struct_format = '10I'  # 'I' represents u32
struct_size = struct.calcsize(struct_format)
sel = selectors.DefaultSelector()

SF_STATE_LENGTH = 8
SF_ACTION_NUM = 3
SF_NUM = 2

state_dim = SF_STATE_LENGTH * SF_NUM  # State dimension = features per subflow * number of subflows
action_dim = pow(SF_ACTION_NUM, SF_NUM)  # 9 possible actions from combinations of DEC/KEEP/INC for each subflow

# reward_history = [] # For recording rewards to analyze convergence

time_step = 0

EPISODE_EQUALS_CONNECTION = False  # True: one episode per connection, False: use fixed EPISODE_LEN

# Episode length and connection reset settings
TRAIN_TIMESTEP_COUNT = 1 # Train every n steps
CON_DURATION = 500  # Connection reset cycle when EPISODE_EQUALS_CONNECTION = False
EPISODE_LEN = 300 if not EPISODE_EQUALS_CONNECTION else CON_DURATION  # Use connection cycle as episode length if alignment enabled

# Print environment configuration
print(f"Environment configuration:")
print(f"Training episode length (EPISODE_LEN): {EPISODE_LEN}")
print(f"Connection reset cycle (CON_DURATION): {CON_DURATION}")
print(f"Training update interval (TRAIN_TIMESTEP_COUNT): {TRAIN_TIMESTEP_COUNT}")
print(f"Episode-connection alignment mode: {'Enabled' if EPISODE_EQUALS_CONNECTION else 'Disabled'}")

PHASE_COLLECT_INITIAL = 0
PHASE_COLLECT_FEEDBACK = 1
current_phase = PHASE_COLLECT_INITIAL

ACTION_MAX= 10

TOTAL_REWARD_MIN = -210
TOTAL_REWARD_MAX = 210

prev_state = None
prev_state_norm = None
prev_action = None

initial_data = {}  # Store subflow data until we have data from both subflows
feedback_data = {}  # Store environment feedback data, two at a time

client_fds = {}  # Key: fd number, Value: source IP

REF_LT_DATA_COUNT = 30
con_init_flag = 1

tput_history = {} # List length is 3
rtt_history = {}

initial_phase_end = 0
feedback_phase_end = 0

last_loop_time = None

iperf3_pid = None

con_reset_permission = 0

CWND_BOUND = (30, 2500)

EXPLORE_INTERVAL = 6  # Exploration interval steps
last_inc_steps = {}
force_dec_flag = {}
is_random_action = False  # Flag for current action randomness

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

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
# device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")
print("============================================================================================")

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
    'hidden_dim':128,  # Not actually used in DTQN
    'n_states': state_dim,
    'n_actions': action_dim,
    'target_update_count': 10000,  # Target network update frequency
    'model_save_interval': 50000,  # Model save interval
   
    # DTQN specific parameters
    'history_len': 20,        # History length for capturing long-term dependencies
    'context_len': 20,        # Context length
    'bag_size': 0,           # Bag size, 0 disables bag mechanism
    'max_env_steps': EPISODE_LEN,  # Maximum environment steps, follows EPISODE_LEN
    'use_dual_flow': True,   # Use dual subflow model
   
    # Prioritized experience replay parameters
    'use_per': False,         # Enable prioritized experience replay
    'per_alpha': 0.6,        # Priority exponent
    'per_beta_start': 0.1,   # Importance sampling initial beta
    'per_beta_frames': 80000, # Frames for beta to reach 1.0
    'per_eps': 1e-6,         # Small value added to TD error to ensure non-zero priority
   
    # DTQN network structure parameters
    'embed_per_obs_dim': 1,  # Embedding size per observation dimension
    'action_embed_dim': 0,   # Action embedding dimension
    'inner_embed_size': 128,  # Inner embedding size
    'num_heads': 4,          # Number of attention heads
    'num_transformer_layers': 1,         # Number of transformer layers
    'dropout': 0.0,          # Dropout ratio
    'gate': "res",           # Use residual gating
    'identity': False,       # Use Post-LN structure (False) or Pre-LN structure (True)
    'pos': "learned",        # Position encoding
    'discrete': False,       # Continuous observation space
}

STATE_HISTORY_LEN = cfg['context_len']+5 # Keep recent states for trend calculation
state_history = deque(maxlen=STATE_HISTORY_LEN)

# Initialize model and memory
model_path = "saved_models/dtqn_model_latest.pth"

if os.path.exists(model_path):
    dtqn_adapter = DTQN_adapter(cfg)
    try:
        dtqn_adapter.load_model(model_path)
        print(f"Loaded pretrained model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}, initializing new model.")
        dtqn_adapter = DTQN_adapter(cfg)
else:
    dtqn_adapter = DTQN_adapter(cfg)

# Add visualization tool
visualizer = DTQNVisualizer(dtqn_adapter)

ENABLE_IPERF_MANAGEMENT = False  # iperf3 management switch
# Enable visualization
ENABLE_VISUALIZATION = False
# Enable separate CSV reward logging
ENABLE_CSV_REWARD_LOGGING = False 

# Enable minimal visualization mode for Transformer Heatmap only
ENABLE_MINIMAL_HEATMAP_ONLY = False # Set to True to only generate heatmaps periodically

# Visualization interval (steps)
VISUALIZATION_INTERVAL = 10000  # Save visualization results every 10000 steps


def calculate_threshold(min_rtt, base_product=9500, growth_factor=0.003, min_rtt_range=(700, 20000)):
    """
    Parameters of this function can be adjusted to adapt to optimal BDP points in different network environments or to meet specific requirements (e.g., low latency), with future work focusing on more accurate delay modeling.
    
    Calculate dynamic threshold as: thresh × min_rtt = base_product × (1 + growth_factor × (min_rtt - min_rtt_range[0])/1000)
    
    Parameters:
        min_rtt (float): Current minimum RTT (ms)
        base_product (float): Base product value (default 9500)
        growth_factor (float): Growth intensity coefficient (default 0.003)
        min_rtt_range (tuple): Valid min_rtt range (default 700-20000ms)
        
    Returns:
        float: Calculated threshold
    """
    min_rtt = np.clip(min_rtt, *min_rtt_range)
    normalized_rtt = (min_rtt - min_rtt_range[0]) / 100  # Normalize per 100ms
    dynamic_product = base_product * (1 + growth_factor * normalized_rtt)
    return dynamic_product / min_rtt
    
def compute_reward(next_state, prev_state, prev_action, client_fds, history_states=None):
    SF_WEIGHT = (1, 1)
    CONTEXT_WINDOW = cfg['context_len']+1 # Window size for trend calculation

    # Convert tensors to numpy arrays to avoid GPU tensor access issues
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
        # Check if there's sufficient variation to calculate slope
        if len(np.unique(data_series)) < 2:
             return 0.0 # If all values are the same, slope is 0
        try:
            slope = np.polyfit(time_axis, data_series, 1)[0]
            return slope
        except (np.linalg.LinAlgError, ValueError):
             # Handle fitting failures
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

        # --- Start: Subflow reward calculation preparation ---
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
        curr_loss_rate = next_state_np[6 + base_idx] 
        prev_loss_rate = prev_state_np[6 + base_idx] 
        explore_flag = prev_state_np[5 + base_idx]
        action = temp_tuple[subflow_index]

        # --- Calculate history-based trends (if history is long enough) ---
        rtt_trend = 0.0
        tput_trend = 0.0

        # Build sequence for trend calculation (time order: oldest -> ... -> prev_state -> next_state)
        rtt_values_in_window = []
        tput_values_in_window = []

        # 1. Get states from history_states
        if CONTEXT_WINDOW >= 3: # Need to get at least one state from history_states
            num_needed_from_history = CONTEXT_WINDOW - 2
            if history_states and len(history_states) >= num_needed_from_history:
                rtt_values_in_window.extend([s[2 + base_idx] for s in history_states[-num_needed_from_history:]])
                tput_values_in_window.extend([s[0 + base_idx] for s in history_states[-num_needed_from_history:]])

        # 2. Add prev_state's rtt and tput (if TREND_WINDOW >= 2)
        if CONTEXT_WINDOW >= 2:
            # prev_rtt and prev_tput already extracted from prev_state
            rtt_values_in_window.append(prev_rtt)
            tput_values_in_window.append(prev_tput)
       
        # 3. Add next_state's rtt and tput (if TREND_WINDOW >= 1)
        if CONTEXT_WINDOW >= 1:
            # curr_rtt and curr_tput already extracted from next_state
            rtt_values_in_window.append(curr_rtt)
            tput_values_in_window.append(curr_tput)
       
       
        # Calculate trends (need at least 2 points)
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

        # Calculate weighted averages of RTT and throughput in the window (more recent samples have higher weights)
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

        # --- End: Subflow reward calculation preparation ---
        # --- Reward calculation logic ---
        rtt_thresh_weight = 0.5
        sf_strong_punish = 0
        sf_reward = 0

        # Modified boundary penalty logic
        if any_sf_hit_boundary:
            sf_strong_punish = -99
            # if curr_cwnd <= CWND_BOUND[0]:
            #     sf_strong_punish = -99 # If any subflow hits boundary, all subflows are penalized
            # else:
            #     sf_strong_punish = -39
            sf_reward = 0
        elif prev_any_sf_hit_boundary and not any_sf_hit_boundary:
            # If previous state hit boundary but current state recovered, give all subflows a reward of 99
            sf_strong_punish = 99
            sf_reward = 0
        elif (explore_flag == 1 and action != 1) or (explore_flag == 2 and action != 2):
            # If explore_flag=1 but this subflow's action is not increase, reward is -99
            sf_strong_punish = -99
            sf_reward = 0
        else:
            # BDP = curr_measured_bw * curr_min_rtt if curr_min_rtt > 0 else 0

            # # 1. BDP-related reward (based on prev_state)
            # if (curr_cwnd < 2 * BDP) and (curr_loss_rate == 0):
            #      # Maintain original logic, encourage increasing window when BDP not reached
            #     #  sf_reward = (prev_cwnd - 2 * BDP) + (curr_cwnd - prev_cwnd) / 4
            #      sf_reward = (curr_cwnd - 2 * BDP)
            # else:
                # 2. RTT and throughput trade-off (combining current state and historical trends)
                RTT_WARNING_THRESH = calculate_threshold(curr_min_rtt) # Use prev_min_rtt to calculate threshold
                k = 6 # Assume no packet loss

                rtt_ratio = avg_rtt_in_window / curr_min_rtt if curr_min_rtt > 0 else 1.0
                alpha = 1 / (1 + math.exp(-k * (rtt_ratio - RTT_WARNING_THRESH)))

                # RTT penalty
                rtt_state_penalty = -rtt_thresh_weight * (rtt_ratio - RTT_WARNING_THRESH) / 0.05
                rtt_trend_penalty = -0 * rtt_trend # Penalize positive slope (RTT increase), coefficient is empirical
                rtt_penalty = rtt_state_penalty + rtt_trend_penalty
                # rtt_penalty = rtt_state_penalty
                # print(f"rtt_state_penalty:{rtt_state_penalty}, rtt_trend_penalty:{rtt_trend_penalty}, rtt_penalty:{rtt_penalty}")
                # print(f"rtt_penalty:{rtt_penalty}")

                # Throughput reward
                tput_state_reward = avg_tput_in_window / 0.001 # Based on previous throughput
                tput_trend_reward = 0 * tput_trend # Reward positive slope (throughput increase), coefficient 60 is empirical
                tput_reward = tput_state_reward + tput_trend_reward
                # tput_reward = tput_state_reward
                # print(f"tput_state_reward:{tput_state_reward}, tput_trend_reward:{tput_trend_reward}, tput_reward:{tput_reward}")
                # print(f"tput_reward:{tput_reward}")

                # Dynamic weighted combination
                sf_reward = alpha * rtt_penalty + (1 - alpha) * tput_reward

        # Combine final reward
        final_sf_reward = SF_WEIGHT[subflow_index] * sf_reward + sf_strong_punish
        return final_sf_reward

    # --- Calculate rewards and return ---
    for i in range(2):
        sf_reward_val = calculate_sf_reward(i, history_states) # Pass history states
        reward_list.append(sf_reward_val)

    total_reward = sum(reward_list)

    # Ensure returning rewards for both subflows and total reward
    # sf1_r = reward_list[0] if len(reward_list) > 0 else 0.0
    # sf2_r = reward_list[1] if len(reward_list) > 1 else 0.0
    sf1_r = reward_list[0]
    sf2_r = reward_list[1]
    return sf1_r, sf2_r, total_reward

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def process_state(state_dic, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step):
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
        # loss_rate = tput_history[ip][2] / tput_history[ip][1]
        loss_rate = 0 # Packet loss not considered in current implementation, reserved

        measured_bw = state_dic[ip][5] / (2 ** 24)
        min_rtt = state_dic[ip][4]
        cwnd = state_dic[ip][6]

        avg_tput_norm = (avg_tput - tput_min) / (tput_max - tput_min)
        measured_bw_norm = (measured_bw - tput_min) / (tput_max - tput_min)
        avg_rtt_norm = (avg_rtt - rtt_min) / (rtt_max - rtt_min)
        min_rtt_norm = (min_rtt - rtt_min) / (rtt_max - rtt_min)
        cwnd_norm = (cwnd - cwnd_min) / (cwnd_max - cwnd_min)
        loss_rate_norm = (loss_rate - loss_min) / (loss_max - loss_min)

        avg_tput_norm = clamp(avg_tput_norm, 0, 1)
        measured_bw_norm = clamp(measured_bw_norm, 0, 1)
        avg_rtt_norm = clamp(avg_rtt_norm, 0, 1)
        min_rtt_norm = clamp(min_rtt_norm, 0, 1)
        cwnd_norm = clamp(cwnd_norm, 0, 1)
        loss_rate_norm = clamp(loss_rate_norm, 0, 1)

        # Test code (commented out)
        # avg_tput_norm = 0
        # measured_bw_norm = 0
        # avg_rtt_norm = 0
        # min_rtt_norm = 0
        # loss_rate_norm = 0

        # avg_tput_norm = int(avg_tput / 0.0005)
        # avg_rtt_norm = int(avg_rtt / 100)       
        # max_tput_norm = int(max_tput / 0.0005)
        # min_rtt_norm = int(min_rtt / 100)
        if (loss_rate > 0) or (avg_rtt/min_rtt >= calculate_threshold(min_rtt)):
            last_inc_steps[ip] = time_step # Delay bandwidth exploration on packet loss

        time_since_last_inc = time_step - last_inc_steps.get(ip, EXPLORE_INTERVAL*10)
        # print(f"time_step:{time_step}, last_inc_steps:{last_inc_steps.get(ip, EXPLORE_INTERVAL*10)}, time_since_last_inc:{time_since_last_inc}")

        if (force_dec_flag[ip] == 666): # Force decrease flag - currently disabled
            explore_flag = 2.0
        elif (time_since_last_inc > EXPLORE_INTERVAL) and (cwnd < (CWND_BOUND[1]-25)):      
            explore_flag = 1.0  # Need to execute INC
        else:
            explore_flag = 0.0

        explore_flag_norm = explore_flag / 2

        # loss_rate and DEMAND_TYPE are reserved but not used
        subflow_state = [avg_tput, measured_bw, avg_rtt, min_rtt, cwnd, explore_flag, loss_rate, DEMAND_TYPE / 2]
        subflow_state_norm = [avg_tput_norm, measured_bw_norm, avg_rtt_norm, min_rtt_norm, cwnd_norm, explore_flag_norm, loss_rate_norm, DEMAND_TYPE / 2]
        # subflow_state_norm = [avg_tput_norm, max_tput_norm, avg_rtt_norm, min_rtt_norm, cwnd]
        # print(f"subflow_state:{subflow_state}, subflow_state_norm:{subflow_state_norm}")
        state.extend(subflow_state)
        state_norm.extend(subflow_state_norm)
       
    state = torch.tensor(state, dtype=torch.float32, device=dtqn_adapter.device)
    state_norm = torch.tensor(state_norm, dtype=torch.float32, device=dtqn_adapter.device)
    return state, state_norm 

def collect_tput_rtt(recv_data, ip, tput_history, rtt_history):
    """Update throughput and RTT data"""
    if ip not in tput_history:
        tput_history[ip] = [0, 0, 0] # total delivered_lt, interval_us_lt, losses_lt
    if ip not in rtt_history:
        rtt_history[ip] = []
    if (recv_data[ip][2] != 0):
        tput_history[ip][0] = recv_data[ip][1]
        tput_history[ip][1] = recv_data[ip][2]
        tput_history[ip][2] = recv_data[ip][0]
    rtt_sample = recv_data[ip][3]
    rtt_history[ip].append(rtt_sample)

def collect_and_initialize_data(data_dict, ip_list):
    """Collect data and initialize related variables"""
    global tput_history, rtt_history, last_inc_steps, force_dec_flag, time_step
   
    for ip in ip_list:
        collect_tput_rtt(data_dict, ip, tput_history, rtt_history)
        if ip not in last_inc_steps:
            if time_step > 0:
                last_inc_steps.setdefault(ip, time_step)
            else:
                last_inc_steps.setdefault(ip, EXPLORE_INTERVAL*10)
        if ip not in force_dec_flag:
            force_dec_flag.setdefault(ip, 0)

def process_state_and_compute_reward(data_dict, data_type):
    """Process state and compute reward"""
    global prev_state, prev_state_norm, prev_action, last_loop_time, tput_history, rtt_history, state_history
   
    next_state, next_state_norm = process_state(data_dict, data_type, tput_history, rtt_history, force_dec_flag, last_inc_steps, time_step)
   
    if not (data_type == 0 and con_init_flag):
        # Get current history state list (not including next_state)
        current_history = list(state_history) # Contains states up to the one *before* prev_state
        sf1_reward, sf2_reward, total_reward = compute_reward(next_state, prev_state, prev_action, client_fds, current_history) # Pass history
       
        # Record time interval
        current_time = time.time()
        if last_loop_time is not None:
            loop_interval = (current_time - last_loop_time) * 1000
            print(f"time_step interval: {loop_interval:.1f}ms")
        last_loop_time = current_time
       
        # After calculating reward, add prev_state (state that led to current reward) to history
        # Note: We store the unnormalized prev_state
        if prev_state is not None:
             state_history.append(prev_state.cpu().numpy()) # Store numpy array
       
        return next_state, next_state_norm, sf1_reward, sf2_reward, total_reward
    else:
        # Also record in initial state
        if next_state is not None:
             state_history.append(next_state.cpu().numpy())
        return next_state, next_state_norm, None, None, None

def print_state_info(data_dict, next_state, prev_state, sf_reward_list, action_tuple):
    """Print state information"""
    global is_random_action
    
    # Convert tensor to numpy array to avoid GPU tensor access issues
    if torch.is_tensor(next_state):
        next_state_np = next_state.cpu().numpy()
    else:
        next_state_np = next_state
        
    if torch.is_tensor(prev_state):
        prev_state_np = prev_state.cpu().numpy()
    else:
        prev_state_np = prev_state
   
    for i, ip in enumerate(sorted(data_dict.keys())):
        action_str = "inc" if action_tuple[i] == 1 else ("hold" if action_tuple[i] == 0 else "dec")
        # Add random action marker
        action_type = "[Random]" if is_random_action else "[Inference]"
        source_ip = socket.inet_ntoa(struct.pack('<I', ip))
       
        # Print state information
        color_code = "\033[33m" if i % 2 == 0 else ""
        color_reset = "\033[0m" if i % 2 == 0 else ""
        print(f"{color_code}source_ip:{source_ip}, cwnd:{prev_state_np[4+i*SF_STATE_LENGTH]}, tput:{prev_state_np[0+i*SF_STATE_LENGTH] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, max_tput:{prev_state_np[1+i*SF_STATE_LENGTH] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, rtt:{int(prev_state_np[2+i*SF_STATE_LENGTH])}, minrtt:{prev_state_np[3+i*SF_STATE_LENGTH]}, loss_rate:{prev_state_np[6+i*SF_STATE_LENGTH]}, exp_flag:{prev_state_np[5+i*SF_STATE_LENGTH]}{', step:'+str(time_step) if i % 2 == 0 else ''}{color_reset}")
        print(f"{color_code}source_ip:{source_ip}, cwnd:{next_state_np[4+i*SF_STATE_LENGTH]}, tput:{next_state_np[0+i*SF_STATE_LENGTH] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, max_tput:{next_state_np[1+i*SF_STATE_LENGTH] * 1e6 * 1500 * 8 / 1e9:.2f}Gbps, rtt:{int(next_state_np[2+i*SF_STATE_LENGTH])}, minrtt:{next_state_np[3+i*SF_STATE_LENGTH]}, loss_rate:{next_state_np[6+i*SF_STATE_LENGTH]}, action:{action_str}{action_type},reward:{sf_reward_list[i]}{color_reset}")
   
        # Check if forced reduction is needed
        if force_dec_flag[ip] != 1 and action_tuple[i] == 1 and (prev_state is not None) and (prev_state_np[i*SF_STATE_LENGTH + 5] == 1) and ((next_state_np[i*SF_STATE_LENGTH + 0] - prev_state_np[i*SF_STATE_LENGTH + 0]) <= 0):
            force_dec_flag[ip] = 1

def update_dqn_and_memory(total_reward, next_state_norm, sf1_reward=None, sf2_reward=None):
    """Update DQN and experience replay memory"""
    global time_step, prev_state, prev_state_norm, prev_action, start_time, con_reset_permission
   
    # Determine if current step marks the end of an episode
    if EPISODE_EQUALS_CONNECTION:
        # When alignment mode is enabled, only CON_DURATION is used as the episode end marker
        current_step_done_float = 1.0 if (time_step % CON_DURATION == CON_DURATION - 1) else 0.0
    else:
        # When alignment mode is disabled, use original condition: either EPISODE_LEN or CON_DURATION satisfied
        current_step_done_float = 1.0 if (time_step % EPISODE_LEN == EPISODE_LEN - 1) or (time_step % CON_DURATION == CON_DURATION - 1) else 0.0
   
    current_step_done_tensor = torch.as_tensor(current_step_done_float, device=dtqn_adapter.device)
   
    total_reward = clamp(total_reward, -600, 600)
    total_reward_norm = (total_reward - TOTAL_REWARD_MIN) / (TOTAL_REWARD_MAX - TOTAL_REWARD_MIN) * 2 - 1
    total_reward_norm = torch.as_tensor(clamp(total_reward_norm, -1.0, 1.0), dtype=torch.float32, device=dtqn_adapter.device)
    print(f"norm reward:{total_reward_norm}")
    # Use observe method to add conversion to DTQN agent's context
    dtqn_adapter.observe(prev_state_norm, prev_action, reward=total_reward_norm, next_state=next_state_norm, done=current_step_done_tensor)
   
    # If current step marks the end of an episode
    if current_step_done_float > 0.0:
        print(f"====== Episode completed ======")
        print(f"Current step count: {time_step}")
        print(f"Episode length: {EPISODE_LEN if not EPISODE_EQUALS_CONNECTION else CON_DURATION}")
        print(f"=========================")
        dtqn_adapter.dtqn_agent.replay_buffer.flush() # Explicitly call flush to end current buffer episode
        dtqn_adapter.prev_state = None # Prepare for context_reset for next episode

    # ---- CSV recording logic ----
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
                current_beta, # Beta value first recorded
                f"{elapsed_t:.2f}" if elapsed_t is not None else "" # Then is Elapsed Time
            ]
            with open(REWARD_CSV_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error writing reward to CSV {REWARD_CSV_FILENAME}: {e}")
    # ---- End CSV recording logic ----
   
    # Display buffer status
    buffer = dtqn_adapter.dtqn_agent.replay_buffer
    print(f"Buffer status - pos: {buffer.pos}, episodes: {buffer.pos[0]}")
   
    # If visualization is enabled, record step data
    if ENABLE_VISUALIZATION:
        if not ENABLE_MINIMAL_HEATMAP_ONLY: # Only record full data in non-minimal mode
            elapsed_time = time.time() - start_time if time_step % 10 == 0 else None
            visualizer.record_step(
                state=prev_state_norm, 
                action=prev_action.item(), 
                reward=total_reward_norm.item(), 
                epsilon=dtqn_adapter.epsilon,
                sf1_reward=sf1_reward.item() if torch.is_tensor(sf1_reward) else sf1_reward,
                sf2_reward=sf2_reward.item() if torch.is_tensor(sf2_reward) else sf2_reward,
                elapsed_time=elapsed_time
            )
        # In minimal mode, no data is recorded, directly get latest weights from transformer layer
   
    time_step += 1
    con_reset_permission = 1 # Restore this line
    
    # Add sleep to reduce action frequency (for performance testing)
    # time.sleep(0.7)  # This value can be adjusted to control action frequency, unit: seconds
   
    # Update model every TRAIN_TIMESTEP_COUNT steps
    if ((time_step != 0) and (time_step % TRAIN_TIMESTEP_COUNT == 0)):
        # Directly try to update model, let adapter handle internal check logic
        print(f"Starting model update, time step: {time_step}, batch_size: {dtqn_adapter.batch_size}")
        start = time.perf_counter()
        train_success = dtqn_adapter.update()
        update_time = (time.perf_counter() - start) * 1000
       
        if train_success:
            print(f"Model update completed: {update_time:.2f} ms, training successful! Current total training count: {dtqn_adapter.train_count}")
            print(f"buffer sample count: {dtqn_adapter.dtqn_agent.replay_buffer.pos[0]}")
        else:
            print(f"Model update completed: {update_time:.2f} ms, training skipped - buffer sample insufficient")
            print(f"buffer sample count: {dtqn_adapter.dtqn_agent.replay_buffer.pos[0]}")
   
    # Generate periodic visualization
    if (time_step != 0):
        if ENABLE_MINIMAL_HEATMAP_ONLY and (time_step % VISUALIZATION_INTERVAL == 0):
            try:
                visualizer.save_attention_weights(step_idx=time_step)
                print(f"Transformer Heatmap generated at step {time_step}")
            except Exception as e:
                print(f"Error generating Transformer Heatmap: {e}")
        elif ENABLE_VISUALIZATION and (time_step % VISUALIZATION_INTERVAL == 0): # Original full visualization logic
            try:
                visualizer.plot_reward_history(window_size=1000)
                visualizer.plot_action_distribution()
                visualizer.plot_epsilon_decay()
               
                # Visualize attention weights and Bag contents
                visualizer.save_attention_weights(step_idx=time_step)
               
                # If bag mechanism is enabled, visualize bag contents
                if hasattr(dtqn_adapter.dtqn_agent, 'bag') and dtqn_adapter.dtqn_agent.bag_size > 0:
                    visualizer.visualize_bag_contents(step_idx=time_step)
               
                # Export historical data
                visualizer.export_history_to_csv()
               
                print(f"Visualization results generated at step {time_step}")
            except Exception as e:
                print(f"Error generating visualization: {e}")

def prepare_next_action(next_state, next_state_norm, data_type):
    """Prepare next action"""
    global prev_state, prev_state_norm, prev_action, tput_history, rtt_history, con_init_flag
    global initial_phase_end, feedback_phase_end, is_random_action
   
    prev_state, prev_state_norm = next_state, next_state_norm
   
    # Process historical data based on data type
    if data_type == 0:
        if not con_init_flag:
            tput_history = {key: [0, 0, 0] for key in tput_history}
            rtt_history = {key: [] for key in rtt_history}
        else:
            con_init_flag = 0
    else:
        tput_history = {key: [0, 0, 0] for key in tput_history}
        rtt_history = {key: [] for key in rtt_history}
   
    action, is_random = dtqn_adapter.sample_action(prev_state_norm)
    action_tuple = action_mapping[action]
    is_random_action = is_random  # Save random action marker
    prev_action = torch.tensor(action, device=dtqn_adapter.device, dtype=torch.long)
   
    # Set phase end flags
    if data_type == 0:
        initial_phase_end = 1
        feedback_phase_end = 0
    else:
        initial_phase_end = 0
        feedback_phase_end = 1
   
    return action_tuple

def send_actions_to_subflows(data_dict, action_tuple, data_type, kernel_data_reset):
    """Send actions to subflows"""
    global last_inc_steps, force_dec_flag, prev_state
    
    # Convert tensor to numpy array to avoid GPU tensor access issues
    if torch.is_tensor(prev_state):
        prev_state_np = prev_state.cpu().numpy()
    else:
        prev_state_np = prev_state
   
    for i, ip in enumerate(sorted(data_dict.keys())):
        if action_tuple[i] == 0:
            desired_cwnd = data_dict[ip][6]
        elif action_tuple[i] == 1:
            desired_cwnd = data_dict[ip][6] + ACTION_MAX
            last_inc_steps[ip] = time_step  # Record growth step
        elif action_tuple[i] == 2:
            # desired_cwnd = data_dict[ip][6] - (2 * ACTION_MAX if prev_state_np[i*SF_STATE_LENGTH + 6] > 0 else ACTION_MAX)
            desired_cwnd = data_dict[ip][6] - ACTION_MAX
            if prev_state is not None and prev_state_np[i*SF_STATE_LENGTH + 5] == 2:
                force_dec_flag[ip] = 0
       
        desired_cwnd = clamp(desired_cwnd, CWND_BOUND[0], CWND_BOUND[1])
       
        response = struct.pack('3I', data_type, kernel_data_reset, desired_cwnd)
        for key in sel.get_map().values():
            if key.fileobj.fileno() in client_fds.keys():
                if client_fds[key.fileobj.fileno()] == ip:
                    try:
                        key.fileobj.sendall(response)
                    except socket.error as e:
                        print(f"Failed to send action: {e}")
                        sel.unregister(key.fileobj)
                        key.fileobj.close()
                    break

def process_data_and_take_action(data_type, data_dict, client_sock):
    """Process data and take action"""
    global time_step, initial_data, feedback_data, current_phase, client_fds, prev_state, prev_state_norm
    global prev_action, tput_history, rtt_history, initial_phase_end, feedback_phase_end
    global last_loop_time, con_init_flag, action_tuple, force_dec_flag, last_inc_steps, con_reset_permission # Add con_reset_permission
   
    # Set phase transition
    next_phase = PHASE_COLLECT_FEEDBACK if data_type == 0 else PHASE_COLLECT_INITIAL
   
    action_tuple = (0, 0)
   
    if len(data_dict) == SF_NUM:
        # Collect data and initialize
        collect_and_initialize_data(data_dict, sorted(data_dict.keys()))
       
        # Check if state should be processed
        should_process = ((data_type == 0 and (feedback_phase_end and (rtt_history and (min(len(values) for values in rtt_history.values()) >= REF_LT_DATA_COUNT)) or con_init_flag)) or 
                          (data_type == 1 and initial_phase_end and (rtt_history and (min(len(values) for values in rtt_history.values()) >= REF_LT_DATA_COUNT))))
       
        kernel_data_reset = 0
       
        if should_process:
            # Process state and compute reward
            next_state, next_state_norm, sf1_reward, sf2_reward, total_reward = process_state_and_compute_reward(data_dict, data_type)
           
            if not (data_type == 0 and con_init_flag):
                # Print state information
                temp_tuple = action_mapping[prev_action.item()]
                print_state_info(data_dict, next_state, prev_state, [sf1_reward, sf2_reward], temp_tuple)
               
                # Update DQN and experience replay memory
                update_dqn_and_memory(total_reward, next_state_norm, sf1_reward, sf2_reward)
           
            # Prepare next action
            action_tuple = prepare_next_action(next_state, next_state_norm, data_type)
           
            kernel_data_reset = 1
       
        # Send actions to subflows
        send_actions_to_subflows(data_dict, action_tuple, data_type, kernel_data_reset)
       
        # Enter next phase
        current_phase = next_phase
       
        # Clear data
        data_dict.clear()
       
        # If connection needs to be reset (using CON_DURATION as connection reset cycle) and IPERF management is enabled
        if ENABLE_IPERF_MANAGEMENT and data_type == 1 and (time_step > 0 and time_step % CON_DURATION == 0) and con_reset_permission:
            if EPISODE_EQUALS_CONNECTION:
                # When alignment mode is enabled, connection can be reset directly because episode length equals connection cycle
                print(f"Connection reset condition met, starting MPTCP connection reset. Current time_step: {time_step}")
                reset_mptcp_connection()
            else:
                # When alignment mode is disabled, ensure current episode has completed
                if time_step % EPISODE_LEN != 0:
                    # If current episode hasn't completed, wait for it to finish
                    print(f"Waiting for current episode to finish before resetting connection. Current time_step: {time_step}, EPISODE_LEN: {EPISODE_LEN}, CON_DURATION: {CON_DURATION}")
                    print(f"Estimated reset in {EPISODE_LEN - (time_step % EPISODE_LEN)} steps")
                else:
                    print(f"Connection reset condition met, starting MPTCP connection reset. Current time_step: {time_step}")
                    reset_mptcp_connection()

def handle_client_data(client_sock):
    global time_step, initial_data, feedback_data, current_phase, client_fds, prev_state, prev_state_norm, prev_action

    try:
        data = client_sock.recv(struct_size)
        if data:
            if len(data) == struct_size:
                unpacked_data = struct.unpack(struct_format, data)
                data_type = unpacked_data[0]
                data_content = unpacked_data[1:-2]  # 0th position is data type, last two positions are congestion control mode and ip address
                if client_fds[client_sock.fileno()] is None:
                    source_ip_int = unpacked_data[-1]  # Get source ip address in integer form
                    client_fds[client_sock.fileno()] = source_ip_int  # Update source ip address in client_fds

                # Get current cwnd value for state synchronization response
                current_cwnd = data_content[6] if len(data_content) > 6 else CWND_BOUND[0]

                if data_type == 0:  # 0 represents initial_data
                    # Initial state data
                    # if current_phase != PHASE_COLLECT_INITIAL:
                    #     print(f"Received unexpected initial data, current phase: {current_phase}, fd: {client_sock.fileno()}")
                    #     # Send feedback response (data_type=1) to help client jump to feedback phase
                    #     response = struct.pack('3I', 1, 0, current_cwnd)  # data_type=1, kernel_data_reset=0, cwnd unchanged
                    #     try:
                    #         client_sock.sendall(response)
                    #         print(f"State synchronization response sent to client fd:{client_sock.fileno()}, helped it jump to feedback phase")
                    #     except socket.error as e:
                    #         print(f"Failed to send state synchronization response: {e}")
                    #         close_sf_cc_sock(client_sock)
                    #     return
                    initial_data[client_fds[client_sock.fileno()]] = data_content  # Add received data to dictionary, key is fd number
                    process_data_and_take_action(data_type, initial_data, client_sock)
                   
                elif data_type == 1:  # 1 represents feedback_data
                    # Feedback data after executing action
                    # if current_phase != PHASE_COLLECT_FEEDBACK:
                    #     print(f"Received unexpected feedback data, current phase: {current_phase}")
                    #     # Send initial response (data_type=0) to help client jump to initial phase
                    #     response = struct.pack('3I', 0, 0, current_cwnd)  # data_type=0, kernel_data_reset=0, cwnd unchanged
                    #     try:
                    #         client_sock.sendall(response)
                    #         print(f"State synchronization response sent to client fd:{client_sock.fileno()}, helped it jump to initial phase")
                    #     except socket.error as e:
                    #         print(f"Failed to send state synchronization response: {e}")
                    #         close_sf_cc_sock(client_sock)
                    #     return
                    feedback_data[client_fds[client_sock.fileno()]] = data_content
                    process_data_and_take_action(data_type, feedback_data, client_sock)
                   
                else:
                    print(f"Unknown data_type: {data_type}")
            else:
                print(f"Invalid data size: {len(data)}")
        else:
            print(f"Client disconnected (Client Disconected_0): {client_sock.fileno()}")
            close_sf_cc_sock(client_sock)
    except socket.error as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print(f"Client disconnected (Client Disconected_1): {client_sock.fileno()}")
            close_sf_cc_sock(client_sock)

def reset_mptcp_connection():
    """Reset MPTCP connection"""
    global iperf3_pid
   
    print("Reset connection...")
    # Close existing connection
    shut_down_mptcp_connection()

    # If IPERF management is enabled, restart IPERF
    if ENABLE_IPERF_MANAGEMENT:
        time.sleep(3)
       
        # Re-establish IPERF connection
        print("Restarting IPERF...")
        iperf_command = ['sudo', 'mptcpize', 'run', '-d', 'iperf3', '-c', '192.168.5.6',  '-t', '36000', '-C', 'jazz-1']
        process = subprocess.Popen(iperf_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            # Get and update global IPERF process group ID
            iperf3_pid = os.getpgid(process.pid)
            print(f"New IPERF process group ID: {iperf3_pid}")
        except Exception as e:
            print(f"Failed to get new IPERF process group ID: {e}")
            iperf3_pid = None # Reset to None
            # Consider whether a stronger error handling is needed, such as exiting script
            # sys.exit(1) 
        time.sleep(1) # Give IPERF a bit of startup time
    else:
        print("IPERF management not enabled, skipping IPERF restart step")

def save_reward_to_csv(time_step, sf1_reward, sf2_reward, reward, prev_action, filename="reward_history.csv"):
    # This function has been removed as its functionality has been integrated into visualizer
    pass

def clear_reward_file(filename=REWARD_CSV_FILENAME): # Modify default parameter to use global variable
    """Clear reward file and write header"""
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time Step", "sf1_reward", "sf2_reward", "total_reward", "Learning Rate", "Epsilon", "Action", "Beta", "Elapsed Time (ms)"]) # Adjust order and add units
        print("Reward file cleared and initialized")
    except Exception as e:
        print(f"Failed to clear reward file: {e}")

def close_sf_cc_sock(client_sock):
    """Close client connection and clean up related resources"""
    sf_connection_reset(client_sock)
    if client_sock.fileno() in client_fds:
        del client_fds[client_sock.fileno()]
    sel.unregister(client_sock)
    client_sock.close()

def sf_connection_reset(client_sock):
    global initial_data, feedback_data, current_phase, con_init_flag, tput_history, rtt_history, initial_phase_end, feedback_phase_end, prev_state, prev_state_norm, prev_action, client_fds, action_mapping, last_inc_steps, force_dec_flag, dtqn_adapter, con_reset_permission # Add con_reset_permission

    # If there is previous state and DTQN adapter exists, force mark current episode as done
    if prev_state is not None and dtqn_adapter is not None and hasattr(dtqn_adapter, 'dtqn_agent') and hasattr(dtqn_adapter.dtqn_agent, 'replay_buffer'):
        # Mark the last state as done=True and add it to the buffer (if there is any state)
        if prev_state_norm is not None and prev_action is not None:
            dummy_next_state = torch.zeros_like(prev_state_norm)
            dummy_reward = torch.tensor(0.0, device=dtqn_adapter.device)
            done_tensor = torch.tensor(1.0, device=dtqn_adapter.device)
           
            # Add this conversion to the buffer and mark it as terminal state
            dtqn_adapter.observe(prev_state_norm, prev_action, reward=dummy_reward, next_state=dummy_next_state, done=done_tensor)
           
        # Explicitly call flush to end current episode
        print("Connection reset: force end current episode")
        dtqn_adapter.dtqn_agent.replay_buffer.flush()
        dtqn_adapter.prev_state = None  # Reset adapter's prev_state

    # Reset server state variables
    current_phase = PHASE_COLLECT_INITIAL
    initial_data.clear()
    feedback_data.clear()
    prev_state = torch.zeros(state_dim, dtype=torch.float32, device=dtqn_adapter.device)
    prev_state_norm = torch.zeros(state_dim, dtype=torch.float32, device=dtqn_adapter.device)
    prev_action = torch.zeros(1, dtype=torch.long, device=dtqn_adapter.device)
    # con_init_flag = 1
    tput_history = {}
    rtt_history = {}
    last_inc_steps = {}
    force_dec_flag = {}
    # initial_phase_end = 0
    # feedback_phase_end = 0
    con_reset_permission = 0 # Restore this line

def accept_client(server_sock): # Restoring the function definition
    global client_fds
    client_sock, _ = server_sock.accept()
    # Modify to dictionary form, initialize source ip as None, update when data received
    client_fds[client_sock.fileno()] = None
    print(f"New connection accepted, fd: {client_sock.fileno()}")
    client_sock.setblocking(False)
    sel.register(client_sock, selectors.EVENT_READ, handle_client_data)

def terminate_iperf3_process():
    """Function to terminate IPERF process, only called when ENABLE_IPERF_MANAGEMENT=True"""
    global iperf3_pid
   
    if iperf3_pid:
        try:
            print(f"Attempting to terminate IPERF process group using SIGTERM: {iperf3_pid}")
            os.killpg(iperf3_pid, signal.SIGTERM)
            print(f"SIGTERM sent to IPERF process group: {iperf3_pid}")
            time.sleep(0.5) # Wait a bit
           
            # Nested try-except block, check if process still exists
            try:
                os.killpg(iperf3_pid, 0) # Check if process exists
                print(f"IPERF process group {iperf3_pid} still exists, attempting to force terminate...")
                os.killpg(iperf3_pid, signal.SIGKILL)
                print(f"SIGKILL sent to IPERF process group: {iperf3_pid}")
                time.sleep(0.1) # Wait for SIGKILL to take effect
            except ProcessLookupError:
                print(f"IPERF process group {iperf3_pid} has successfully terminated (or exited on its own).")
            except Exception as e_check:
                print(f"Error checking or force terminating IPERF process group {iperf3_pid}: {e_check}")
               
        except ProcessLookupError:
            print(f"IPERF process group {iperf3_pid} no longer exists before attempting to terminate.")
        except Exception as e:
            print(f"Error terminating IPERF process group {iperf3_pid}: {e}")
        finally:
            iperf3_pid = None # Reset pid
    else:
        print("No valid IPERF process group ID to terminate.")

def clean_connections_with_tcpkill():
    """Function to clean up connections using tcpkill, called regardless of ENABLE_IPERF_MANAGEMENT setting"""
    # Start and stop tcpkill process to clean up residual connections
    print("Starting tcpkill process to clean up connections...")
    tcpkill_pgids = []
    tcpkill_processes = []
    try:
        command_0 = ['sudo', 'tcpkill', '-i', 'wlp1s0', 'host', '192.168.5.3']
        process_0 = subprocess.Popen(command_0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        tcpkill_processes.append(process_0)
        tcpkill_pgids.append(os.getpgid(process_0.pid))
        print(f"tcpkill (wlp1s0) started, PGID: {tcpkill_pgids[-1]}")
    except Exception as e:
        print(f"Failed to start first tcpkill: {e}")
       
    try:
        command_1 = ['sudo', 'tcpkill', '-i', 'wlp2s0', 'host', '192.168.0.66']
        process_1 = subprocess.Popen(command_1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        tcpkill_processes.append(process_1)
        tcpkill_pgids.append(os.getpgid(process_1.pid))
        print(f"tcpkill (wlp2s0) started, PGID: {tcpkill_pgids[-1]}")
    except Exception as e:
        print(f"Failed to start second tcpkill: {e}")

    if tcpkill_pgids:
        print(f"Waiting for {len(tcpkill_pgids)} tcpkill processes to finish...")
        time.sleep(1.5) # Give tcpkill a bit of time to catch and terminate connections
        print("Attempting to terminate tcpkill processes...")
        for pgid in tcpkill_pgids:
            try:
                print(f"   Attempting to terminate tcpkill process group: {pgid}")
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(0.2) # Wait briefly
               
                # Nested try-except block, check if process still exists
                try:
                    os.killpg(pgid, 0) # Check if process exists
                    print(f"  tcpkill process group {pgid} still exists, attempting to force terminate...")
                    os.killpg(pgid, signal.SIGKILL)
                    print(f"   SIGKILL sent to tcpkill process group {pgid}.")
                except ProcessLookupError:
                    print(f"  tcpkill process group {pgid} has successfully terminated.")
                except Exception as e_kill_check:
                    print(f"   Error checking or force terminating tcpkill process group {pgid}: {e_kill_check}")
                   
            except ProcessLookupError:
                print(f"  tcpkill process group {pgid} no longer exists before attempting to terminate.")
            except Exception as e_kill:
                print(f"   Error terminating tcpkill process group {pgid}: {e_kill}")
    else:
        print("No successful tcpkill processes to terminate.")

def reset_client_sockets():
    """Function to reset client socket connections, called regardless of ENABLE_IPERF_MANAGEMENT setting"""
    # Reset state and socket of each client connection on the server side
    print("Resetting client socket connections...")
    client_socks_to_close = []
    if sel: # Ensure sel object exists
        try:
            map_values = sel.get_map().values()
            client_socks_to_close = [key.fileobj for key in map_values
                                    if hasattr(key.fileobj, 'fileno') and key.fileobj.fileno() in client_fds]
        except Exception as e_sel:
            print(f"Error getting selector mapping: {e_sel}") # sel may be cleaned up during program exit

    if client_socks_to_close:
        print(f"Preparing to close {len(client_socks_to_close)} client socket connections...")
        # Create a copy of the list for iteration, as close_sf_cc_sock modifies client_fds
        for sock in list(client_socks_to_close):
             if sock.fileno() in client_fds: # Check again, in case removed during iteration
                 print(f"Closing socket fd: {sock.fileno()}")
                 close_sf_cc_sock(sock)
        print("All active client socket connections closed.")
    else:
        print("No active client socket connections to close.")

def shut_down_mptcp_connection():
    """Function to close MPTCP connection, includes three main steps:
    1. Terminate IPERF process (only if ENABLE_IPERF_MANAGEMENT=True)
    2. Clean up connections using tcpkill (always executed)
    3. Reset client socket connections (always executed)
    """
    # 1. Only terminate IPERF process if IPERF management is enabled
    if ENABLE_IPERF_MANAGEMENT:
        terminate_iperf3_process()
   
    # 2. Always use tcpkill to clean up connections
    clean_connections_with_tcpkill()
   
    # 3. Always reset client socket connections
    reset_client_sockets()

    # 4. Attempt to clean up any residual tcpkill processes (as a last resort)
    # This step is optional, as the above killpg should have handled it
    # try:
    #     print("Attempting to clean up any residual tcpkill processes...")
    #     command_3 = ['sudo', 'pkill', '-f', 'tcpkill'] # Use -f to match full command line, more precise
    #     process_3 = subprocess.Popen(command_3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #     process_3.wait(timeout=2) # Wait for pkill to finish, set timeout
    #     print("pkill cleanup attempt completed.")
    # except FileNotFoundError:
    #      print("pkill command not found, skipping pkill cleanup")
    # except subprocess.TimeoutExpired:
    #     print("pkill cleanup timed out")
    # except Exception as e:
    #     print(f"Error cleaning up tcpkill processes: {e}")

def signal_handler(sig=None, frame=None):
    """Signal handler function, executed when program exits"""
    global client_fd, client_sock, dtqn_adapter, _exit_handler_called
   
    # Avoid repeated calls
    if hasattr(signal_handler, '_called'):
        return
    signal_handler._called = True
   
    print('Forced exit!')
   
    try:
        # Generate analysis report before exiting
        if ENABLE_VISUALIZATION and not ENABLE_MINIMAL_HEATMAP_ONLY: # Only generate report in non-minimal mode
            visualizer.generate_analysis_report()
            visualizer.plot_reward_history()
            visualizer.plot_action_distribution()
            visualizer.export_history_to_csv()
            print("Final analysis report and visualization results generated")
    except Exception as e:
        print(f"Error generating analysis report: {e}")
   
    try:
        # Save final model
        if False:
            model_path = f"dtqn_model_final_{dtqn_adapter.train_count}.pth"
            dtqn_adapter.save_model()
            print(f"Final model saved: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
   
    try:
        # Close connection
        # If IPERF management is enabled, IPERF process will be terminated
        # Regardless, tcpkill and socket reset will be executed
        shut_down_mptcp_connection()
    except Exception as e:
        print(f"Error closing MPTCP connection: {e}")
   
    # Exit program
    sys.exit(0)
   
def main():
    # global iperf3_pid, state_dim, start_time, sel, server_sock # Restore iperf3_pid
    global state_dim, start_time, sel, server_sock # Ensure sel and server_sock are available in finally block
    if ENABLE_IPERF_MANAGEMENT: # Add new condition
        global iperf3_pid # Move to condition block

    # Register signal handler function
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    # Register atexit handler function
    # atexit.register(signal_handler)
    # No longer need atexit registration, as signal_handler can handle all cleanup work
    # atexit.register(signal_handler)

    # Ensure sel is initialized
    sel = selectors.DefaultSelector()
    server_sock = None # Initialize server_sock
    start_time = None # Initialize start_time

    try:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        # Create Unix domain stream socket
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        server_sock.listen(5)
        server_sock.setblocking(False)
       
        sel.register(server_sock, selectors.EVENT_READ, accept_client)

        print(f"Server listening on {SOCKET_PATH}")

        # Model warm-up
        dummy_state = torch.FloatTensor(np.random.rand(state_dim)).to(dtqn_adapter.device)
        with torch.no_grad():
            _ = dtqn_adapter.sample_action(dummy_state)
        print("Model warm-up completed")

        if ENABLE_CSV_REWARD_LOGGING: # Add switch check
            clear_reward_file() # Called with REWARD_CSV_FILENAME
       
        if ENABLE_IPERF_MANAGEMENT: # Add new condition
            print("Starting initial IPERF...") 
            iperf_command = ['sudo', 'mptcpize', 'run', '-d', 'iperf3', '-c', '192.168.5.6',  '-t', '36000', '-C', 'jazz-1'] 
            process = subprocess.Popen(iperf_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True) 
            try:
                # Get initial IPERF process group ID
                iperf3_pid = os.getpgid(process.pid)
                print(f"Initial IPERF process group ID: {iperf3_pid}")
            except Exception as e:
                print(f"Failed to get initial IPERF process group ID: {e}")
                iperf3_pid = None
                # Start failed, may need to exit or take other measures
                sys.exit(1) 

        start_time = time.time() # Correctly initialize start_time
        while True:
            events = sel.select(timeout=None)  # Wait for events to occur
            for key, _ in events:
                callback = key.data
                callback(key.fileobj)
    except Exception as e:  # Catch all exceptions
        print(f"Server encountered error: {e}")
        # raise  # Re-raise exception to trigger atexit or signal_handler
    finally:
        print("Server is shutting down...")
        # Note: If all exceptions are caught in try block and not re-raised,
        # or program exits normally, signal_handler may not be called.
        # Call cleanup function to ensure execution.
        if ENABLE_IPERF_MANAGEMENT: # Add new condition
            signal_handler() # Manually call cleanup 

        # Additional cleanup (in case signal_handler hasn't fully executed or not called)
        if sel:
            try:
                if server_sock:
                    sel.unregister(server_sock)
            except Exception as e_unreg:
                print(f"Error unregistering server socket: {e_unreg}")
            finally:
                sel.close() # Close selector
                print("Selector closed.")
        if server_sock:
            server_sock.close()
            print("Server socket closed.")
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
                print(f"Unix socket file {SOCKET_PATH} deleted.")
            except OSError as e_remove:
                print(f"Error deleting Unix socket file {SOCKET_PATH}: {e_remove}")
        print("Server shutdown completed.")

if __name__ == "__main__":
    main()