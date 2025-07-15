import socket
import os
import selectors
import struct
import errno
import time
import signal
import sys
import atexit
import json
import numpy as np
import threading
from collections import deque
import pickle
import subprocess

# Socket communication related constants
SOCKET_PATH = "/tmp/mysocket"  # Unix Socket for kernel communication
BUFFER_SIZE = 1024
BACKEND_HOST = "192.168.213.1"  # DRL server IP address, modify according to actual situation
BACKEND_PORT = 9000  # DRL server port

# Data format definition
struct_format = '10I'  # 'I' represents u32
struct_size = struct.calcsize(struct_format)
sel = selectors.DefaultSelector()

# Subflow related configuration
SF_STATE_LENGTH = 8
SF_ACTION_NUM = 3
SF_NUM = 2

# State and action dimensions
state_dim = SF_STATE_LENGTH * SF_NUM
action_dim = pow(SF_ACTION_NUM, SF_NUM)

# Global variables
time_step = 0
client_fds = {}  # Keys are fd numbers, values are source IP addresses
tput_history = {}  # Throughput history
rtt_history = {}  # RTT history
initial_data = {}  # Temporary storage for data from subflows
feedback_data = {}  # Temporary storage for environment feedback data

prev_state = None
prev_action = None
last_inc_steps = {}
force_dec_flag = {}

# Phase definitions
PHASE_COLLECT_INITIAL = 0
PHASE_COLLECT_FEEDBACK = 1
current_phase = PHASE_COLLECT_INITIAL

# Other constant definitions
DEMAND_TYPE = 0
CON_DURATION = 500  # Connection reset cycle
EPISODE_EQUALS_CONNECTION = False
EPISODE_LEN = 300 if not EPISODE_EQUALS_CONNECTION else CON_DURATION
TRAIN_TIMESTEP_COUNT = 1
REF_LT_DATA_COUNT = 30
EXPLORE_INTERVAL = 6
ACTION_MAX = 10
CWND_BOUND = (30, 2500)
con_init_flag = 1
con_reset_permission = 0
initial_phase_end = 0
feedback_phase_end = 0

# For iperf connection management
ENABLE_IPERF_MANAGEMENT = False
iperf3_pid = None

# Action mapping
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

# Backend communication class
class BackendClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to backend server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to backend server: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to backend server: {e}")
            self.connected = False
            return False
    
    def send_data(self, data_type, data_dict):
        """Send data to backend"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Serialize data
            data_to_send = {
                "data_type": data_type,
                "state_dict": data_dict,
                "time_step": time_step,
                "tput_history": tput_history,
                "rtt_history": rtt_history,
                "prev_action": prev_action,
                "last_inc_steps": last_inc_steps,
                "force_dec_flag": force_dec_flag,
                "con_init_flag": con_init_flag
            }
            serialized_data = pickle.dumps(data_to_send)
            
            # Send data length
            size = len(serialized_data)
            self.socket.sendall(struct.pack('!I', size))
            
            # Send data
            self.socket.sendall(serialized_data)
            
            # Receive response length
            size_data = self.socket.recv(4)
            if not size_data:
                raise ConnectionError("Failed to receive response length")
            
            response_size = struct.unpack('!I', size_data)[0]
            
            # Receive response data
            chunks = []
            bytes_recvd = 0
            while bytes_recvd < response_size:
                chunk = self.socket.recv(min(response_size - bytes_recvd, 4096))
                if not chunk:
                    raise ConnectionError("Failed to receive response data")
                chunks.append(chunk)
                bytes_recvd += len(chunk)
            
            response_data = pickle.loads(b''.join(chunks))
            return response_data
        except Exception as e:
            print(f"Failed to communicate with backend: {e}")
            self.connected = False
            return None
    
    def close(self):
        """Close connection to backend"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False

# Create backend client instance
backend_client = BackendClient(BACKEND_HOST, BACKEND_PORT)

# Main function definitions
def clamp(value, min_value, max_value):
    """Limit value within specified range"""
    return max(min_value, min(value, max_value))

def collect_tput_rtt(recv_data, ip, tput_history, rtt_history):
    """Update throughput and RTT data"""
    if ip not in tput_history:
        tput_history[ip] = [0, 0, 0]  # total delivered_lt, interval_us_lt, losses_lt
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

def send_actions_to_subflows(data_dict, action_tuple, data_type, kernel_data_reset):
    """Send actions to subflows"""
    global last_inc_steps, force_dec_flag, prev_state
   
    for i, ip in enumerate(sorted(data_dict.keys())):
        if action_tuple[i] == 0:
            desired_cwnd = data_dict[ip][6]
        elif action_tuple[i] == 1:
            desired_cwnd = data_dict[ip][6] + ACTION_MAX
            last_inc_steps[ip] = time_step  # Record growth timestep
        elif action_tuple[i] == 2:
            desired_cwnd = data_dict[ip][6] - ACTION_MAX
       
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
    """Process data and take actions"""
    global time_step, initial_data, feedback_data, current_phase, client_fds, prev_state
    global prev_action, tput_history, rtt_history, initial_phase_end, feedback_phase_end
    global con_init_flag, action_tuple, force_dec_flag, last_inc_steps, con_reset_permission
   
    # Set phase transition
    next_phase = PHASE_COLLECT_FEEDBACK if data_type == 0 else PHASE_COLLECT_INITIAL
   
    action_tuple = (0, 0)
   
    if len(data_dict) == SF_NUM:
        # Collect data and initialize
        collect_and_initialize_data(data_dict, sorted(data_dict.keys()))
       
        # Determine whether to process state
        should_process = ((data_type == 0 and (feedback_phase_end and (rtt_history and (min(len(values) for values in rtt_history.values()) >= REF_LT_DATA_COUNT)) or con_init_flag)) or 
                          (data_type == 1 and initial_phase_end and (rtt_history and (min(len(values) for values in rtt_history.values()) >= REF_LT_DATA_COUNT))))
       
        kernel_data_reset = 0
       
        if should_process:
            # Send data to backend server for processing
            response = backend_client.send_data(data_type, data_dict)
            
            if response:
                # Update frontend state
                prev_state = response.get("next_state")
                prev_action = response.get("action")
                action_tuple = response.get("action_tuple")
                time_step += 1
                
                # Update control flags
                force_dec_flag = response.get("force_dec_flag", force_dec_flag)
                last_inc_steps = response.get("last_inc_steps", last_inc_steps)
                initial_phase_end = response.get("initial_phase_end", initial_phase_end)
                feedback_phase_end = response.get("feedback_phase_end", feedback_phase_end)
                con_init_flag = response.get("con_init_flag", con_init_flag)
                
                kernel_data_reset = 1
            else:
                print("Unable to get backend response, using default action")
       
        # Send actions to subflows
        send_actions_to_subflows(data_dict, action_tuple, data_type, kernel_data_reset)
       
        # Enter next phase
        current_phase = next_phase
       
        # Clear data
        data_dict.clear()
       
        # If connection reset is needed (using CON_DURATION as connection reset cycle) and IPERF management is enabled
        if ENABLE_IPERF_MANAGEMENT and data_type == 1 and (time_step > 0 and time_step % CON_DURATION == 0) and con_reset_permission:
            if EPISODE_EQUALS_CONNECTION:
                # When alignment mode is enabled, can directly reset connection as episode length equals connection cycle
                print(f"Connection reset condition met, starting MPTCP connection reset. Current time_step: {time_step}")
                reset_mptcp_connection()
            else:
                # When alignment mode is not enabled, need to ensure current episode is completed
                if time_step % EPISODE_LEN != 0:
                    # If current episode is not completed, wait for it to complete
                    print(f"Waiting for current episode to complete before resetting connection. Current time_step: {time_step}, EPISODE_LEN: {EPISODE_LEN}, CON_DURATION: {CON_DURATION}")
                    print(f"Expected to reset connection in {EPISODE_LEN - (time_step % EPISODE_LEN)} steps")
                else:
                    print(f"Connection reset condition met, starting MPTCP connection reset. Current time_step: {time_step}")
                    reset_mptcp_connection()

def handle_client_data(client_sock):
    global time_step, initial_data, feedback_data, current_phase, client_fds, prev_state, prev_action

    try:
        data = client_sock.recv(struct_size)
        if data:
            if len(data) == struct_size:
                unpacked_data = struct.unpack(struct_format, data)
                data_type = unpacked_data[0]
                data_content = unpacked_data[1:-2]  # First position is data type, last two are congestion control mode and IP address
                if client_fds[client_sock.fileno()] is None:
                    source_ip_int = unpacked_data[-1]  # Get integer form of source IP address
                    client_fds[client_sock.fileno()] = source_ip_int  # Update source IP in client_fds

                # Get current cwnd value for state synchronization response
                current_cwnd = data_content[6] if len(data_content) > 6 else CWND_BOUND[0]

                if data_type == 0:  # 0 indicates initial_data
                    initial_data[client_fds[client_sock.fileno()]] = data_content
                    process_data_and_take_action(data_type, initial_data, client_sock)
                   
                elif data_type == 1:  # 1 indicates feedback_data
                    feedback_data[client_fds[client_sock.fileno()]] = data_content
                    process_data_and_take_action(data_type, feedback_data, client_sock)
                   
                else:
                    print(f"Unknown data_type: {data_type}")
            else:
                print(f"Invalid data size: {len(data)}")
        else:
            print(f"Client ({client_fds.get(client_sock.fileno(), 'unknown')}) disconnected (no data received)")
            close_sf_cc_sock(client_sock)
    except socket.error as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print(f"Client ({client_fds.get(client_sock.fileno(), 'unknown')}) disconnected due to socket error: {e}")
            close_sf_cc_sock(client_sock)
    except Exception as e:
        print(f"Unexpected error handling client ({client_fds.get(client_sock.fileno(), 'unknown')}): {e}")
        close_sf_cc_sock(client_sock)

def accept_client(server_sock):
    global client_fds
    client_sock, _ = server_sock.accept()
    # Modified to dictionary form, initially set source IP to None, update when data is received
    client_fds[client_sock.fileno()] = None
    print(f"New connection accepted, fd: {client_sock.fileno()}")
    client_sock.setblocking(False)
    sel.register(client_sock, selectors.EVENT_READ, handle_client_data)

def close_sf_cc_sock(client_sock):
    """Close client connection and clean up related resources"""
    print(f"Client disconnected, notifying backend to reset state...")
    
    # Notify backend of connection reset
    try:
        if backend_client.connected:
            reset_data = {
                "prev_state": prev_state,
                "prev_action": prev_action
            }
            response = backend_client.send_data("reset", reset_data)
            if response and response.get("status") == "reset_handled":
                print("Backend state reset successfully")
            else:
                print("Backend state reset failed or no response")
    except Exception as e:
        print(f"Failed to notify backend about connection reset: {e}")
    
    # Reset frontend state
    sf_connection_reset(client_sock)
    
    # Clean up connection mapping
    if client_sock.fileno() in client_fds:
        del client_fds[client_sock.fileno()]
    
    # Unregister and close socket
    try:
        sel.unregister(client_sock)
        client_sock.close()
    except Exception as e:
        print(f"Error closing client socket: {e}")

def sf_connection_reset(client_sock):
    global initial_data, feedback_data, current_phase, con_init_flag, tput_history
    global rtt_history, initial_phase_end, feedback_phase_end, prev_state
    global prev_action, client_fds, action_mapping, last_inc_steps, force_dec_flag, con_reset_permission

    print("Resetting frontend state variables...")
    
    # Reset server state variables
    current_phase = PHASE_COLLECT_INITIAL
    initial_data.clear()
    feedback_data.clear()
    prev_state = None
    prev_action = None
    tput_history = {}
    rtt_history = {}
    last_inc_steps = {}
    force_dec_flag = {}
    con_reset_permission = 0
    
    print("Frontend state reset complete")

def reset_mptcp_connection():
    """Reset MPTCP connection"""
    global iperf3_pid
   
    print("Resetting connection...")
    # Close existing connection
    shut_down_mptcp_connection()

    # If iperf3 management is enabled, restart iperf3
    if ENABLE_IPERF_MANAGEMENT:
        time.sleep(3)
       
        # Re-establish iperf connection
        print("Restarting iperf3...")
        iperf_command = ['sudo', 'mptcpize', 'run', '-d', 'iperf3', '-c', '192.168.5.6',  '-t', '36000', '-C', 'jazz-1']
        process = subprocess.Popen(iperf_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            # Get and update global iperf3 process group ID
            iperf3_pid = os.getpgid(process.pid)
            print(f"New iperf3 process group ID: {iperf3_pid}")
        except Exception as e:
            print(f"Failed to get new iperf3 process group ID: {e}")
            iperf3_pid = None
        time.sleep(1)
    else:
        print("IPERF3 management not enabled, skipping iperf3 restart step")

def terminate_iperf3_process():
    """Terminate iperf3 process"""
    global iperf3_pid
    
    if iperf3_pid:
        try:
            print(f"Attempting to terminate iperf3 process group with SIGTERM: {iperf3_pid}")
            os.killpg(iperf3_pid, signal.SIGTERM)
            print(f"SIGTERM sent to iperf3 process group: {iperf3_pid}")
            time.sleep(0.5)
            
            try:
                os.killpg(iperf3_pid, 0)
                print(f"Process group {iperf3_pid} still exists, attempting to force terminate with SIGKILL...")
                os.killpg(iperf3_pid, signal.SIGKILL)
                print(f"SIGKILL sent to iperf3 process group: {iperf3_pid}")
                time.sleep(0.1)
            except ProcessLookupError:
                print(f"Process group {iperf3_pid} successfully terminated (or exited after SIGTERM).")
            except Exception as e_check:
                print(f"Error checking or force terminating iperf3 process group {iperf3_pid}: {e_check}")
                
        except ProcessLookupError:
            print(f"iperf3 process group {iperf3_pid} no longer exists before termination attempt.")
        except Exception as e:
            print(f"Error terminating iperf3 process group {iperf3_pid}: {e}")
        finally:
            iperf3_pid = None

def clean_connections_with_tcpkill():
    """Clean up connections using tcpkill"""
    print("Starting tcpkill process to clean up connections...")
    tcpkill_pgids = []
    tcpkill_processes = []
    try:
        command_0 = ['sudo', 'tcpkill', '-i', 'wlp1s0', 'host', '192.168.5.3']
        process_0 = subprocess.Popen(command_0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        tcpkill_processes.append(process_0)
        tcpkill_pgids.append(os.getpgid(process_0.pid))
        print(f"Started tcpkill (wlp1s0), PGID: {tcpkill_pgids[-1]}")
    except Exception as e:
        print(f"Failed to start first tcpkill: {e}")
       
    try:
        command_1 = ['sudo', 'tcpkill', '-i', 'wlp2s0', 'host', '192.168.0.66']
        process_1 = subprocess.Popen(command_1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        tcpkill_processes.append(process_1)
        tcpkill_pgids.append(os.getpgid(process_1.pid))
        print(f"Started tcpkill (wlp2s0), PGID: {tcpkill_pgids[-1]}")
    except Exception as e:
        print(f"Failed to start second tcpkill: {e}")

    if tcpkill_pgids:
        print(f"Waiting for {len(tcpkill_pgids)} tcpkill processes to work...")
        time.sleep(1.5)
        print("Attempting to terminate tcpkill processes...")
        for pgid in tcpkill_pgids:
            try:
                print(f"  Attempting to terminate tcpkill process group with SIGTERM: {pgid}")
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(0.2)
                
                try:
                    os.killpg(pgid, 0)
                    print(f"  tcpkill process group {pgid} still exists, attempting to use SIGKILL...")
                    os.killpg(pgid, signal.SIGKILL)
                    print(f"  SIGKILL sent to tcpkill process group {pgid}.")
                except ProcessLookupError:
                    print(f"  tcpkill process group {pgid} successfully terminated.")
                except Exception as e_kill_check:
                    print(f"  Error checking or force terminating tcpkill process group {pgid}: {e_kill_check}")
                    
            except ProcessLookupError:
                print(f"  tcpkill process group {pgid} no longer exists before termination attempt.")
            except Exception as e_kill:
                print(f"  Error terminating tcpkill process group {pgid}: {e_kill}")
    else:
        print("No successfully started tcpkill processes to terminate.")

def reset_client_sockets():
    """Reset client socket connections"""
    print("Resetting client socket connections...")
    client_socks_to_close = []
    if sel:
        try:
            map_values = sel.get_map().values()
            client_socks_to_close = [key.fileobj for key in map_values
                                    if hasattr(key.fileobj, 'fileno') and key.fileobj.fileno() in client_fds]
        except Exception as e_sel:
            print(f"Error getting selector map: {e_sel}")

    if client_socks_to_close:
        print(f"Preparing to close {len(client_socks_to_close)} client socket connections...")
        for sock in list(client_socks_to_close):
             if sock.fileno() in client_fds:
                 print(f"Closing socket fd: {sock.fileno()}")
                 close_sf_cc_sock(sock)
        print("All active client socket connections have been closed.")
    else:
        print("No active client socket connections need to be closed.")

def shut_down_mptcp_connection():
    """Shut down MPTCP connection"""
    if ENABLE_IPERF_MANAGEMENT:
        terminate_iperf3_process()
   
    clean_connections_with_tcpkill()
   
    reset_client_sockets()

def signal_handler(sig=None, frame=None):
    """Signal handler function"""
    if hasattr(signal_handler, '_called'):
        return
    signal_handler._called = True
    
    print('Force exit!')
    
    try:
        if ENABLE_IPERF_MANAGEMENT:
            shut_down_mptcp_connection()
        backend_client.close()
    except:
        pass
    
    sys.exit(0)

def main():
    """Main function"""
    global sel, server_sock
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        
        # Create Unix domain socket
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        server_sock.listen(5)
        server_sock.setblocking(False)
        
        sel.register(server_sock, selectors.EVENT_READ, accept_client)
        
        print(f"Frontend server listening on {SOCKET_PATH}")
        
        # Connect to backend server
        if not backend_client.connect():
            print("Cannot connect to backend server, will retry during data processing")
        
        # Main loop
        while True:
            events = sel.select(timeout=None)
            for key, _ in events:
                callback = key.data
                callback(key.fileobj)
    except Exception as e:
        print(f"Server encountered an error: {e}")
    finally:
        print("Server is shutting down...")
        if sel:
            try:
                if server_sock:
                    sel.unregister(server_sock)
            except:
                pass
            sel.close()
        if server_sock:
            server_sock.close()
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except:
                pass
        backend_client.close()
        print("Server shutdown complete")

if __name__ == "__main__":
    main() 