#!/usr/bin/env python
"""
Ultra‐Intelligent Self‐Aware Chess AI Trainer: Adaptive Self‐Play with Dynamic Hyperparameter Tuning

About:
    This advanced chess AI trainer uses deep learning, Monte Carlo Tree Search (MCTS),
    and multi-threaded minimax search to continuously self-play and learn.
    It is “self-aware” in that it periodically scans the evaluation distribution of
    random chess positions and automatically adjusts hyperparameters (e.g. learning rate,
    exploration rate) if it detects that the model is stuck in a plateau (for example,
    evaluations remain too neutral with a very low standard deviation).
    Robust logging, automated file recovery, and dynamic tuning ensure that the model
    adapts to challenges and continues to improve over time.
"""

#############################
# Import Standard Libraries
#############################
import os                      # For file and OS-level operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # Prevent issues with MKL duplicate libraries

import sys                     # System-specific parameters and functions
import time                    # Timing functions for measuring durations
import math                    # Mathematical functions (e.g. inf, sqrt)
import random                  # Random number generation
import pickle                  # For saving and loading Python objects
import warnings                # To control warning messages
import threading               # For multi-threaded operations
import concurrent.futures      # To run parallel tasks (for minimax search)
import atexit                  # Register cleanup functions at program exit
import psutil                  # For monitoring system memory usage

#############################
# Import Third-Party Libraries
#############################
import numpy as np           # Numerical operations and array handling
import torch                 # PyTorch for deep learning
import torch.nn as nn        # Neural network modules
import torch.optim as optim  # Optimizers for training
import chess                 # Python-chess library for board representation and move generation

#############################
# Utility Functions
#############################
def format_duration(total_seconds):
    """Convert seconds into a human-readable format."""
    years = total_seconds // 31536000
    total_seconds %= 31536000
    months = total_seconds // 2592000
    total_seconds %= 2592000
    days = total_seconds // 86400
    total_seconds %= 86400
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{int(years)}y {int(months)}m {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

def format_file_size(size_bytes):
    """Convert a file size in bytes into a human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024.0
        index += 1
    return f"{size_bytes:,.1f} {units[index]}"

def get_training_mode():
    """Return a string describing the current training device."""
    if device.type == "cuda":
        if use_amp:
            return f"GPU-Tensor (AMP enabled) - {torch.cuda.get_device_name(device)}"
        else:
            return f"GPU-CUDA (FP32) - {torch.cuda.get_device_name(device)}"
    else:
        return f"CPU (using {torch.get_num_threads()} threads)"

#############################
# Global Configurations and Setup
#############################
warnings.filterwarnings("ignore", message=".*dpi.*")
EXTENDED_DEBUG = False  # Set True to enable verbose logging
use_amp = False         # Whether to use Automatic Mixed Precision

# Global master variables for merging model and table data
MASTER_MODEL_RAM = None  
MASTER_TABLE_RAM = {}
master_lock = threading.Lock()

# Global training start time
training_active_start_time = None

#############################
# Hyperparameters & File Paths
#############################
STATE_SIZE = 768      # Board encoding vector size
MOVE_SIZE = 128       # Move encoding vector size
INPUT_SIZE = STATE_SIZE + MOVE_SIZE

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS_PER_GAME = 5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99999

USE_MCTS = True
MCTS_SIMULATIONS = 2000
MCTS_EXPLORATION_PARAM = 1.4

MOVE_TIME_LIMIT = 300.0
INITIAL_CLOCK = 600.0
SAVE_INTERVAL_SECONDS = 60

MODEL_SAVE_PATH_WHITE = "white_dqn.pt"
MODEL_SAVE_PATH_BLACK = "black_dqn.pt"
TABLE_SAVE_PATH_WHITE = "white_transposition.pkl"
TABLE_SAVE_PATH_BLACK = "black_transposition.pkl"
STATS_FILE = "stats.pkl"
MASTER_MODEL_SAVE_PATH = "master_dqn.pt"
MASTER_TABLE_SAVE_PATH = "master_transposition.pkl"

# Self-aware adjustment settings
EVAL_STD_THRESHOLD = 0.001       # If evaluation std-dev falls below this, model is "stuck"
LR_ADJUST_FACTOR = 1.2           # Factor to increase learning rate on adjustment
EPSILON_ADJUST_FACTOR = 1.1      # Factor to increase exploration epsilon
AGGRESSIVE_COUNT_THRESHOLD = 3   # Number of consecutive low-variance scans before aggressive adjustment

#############################
# Device Selection
#############################
if not torch.cuda.is_available():
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"Optimized for CPU: Using {torch.get_num_threads()} threads.")
    device = torch.device("cpu")
else:
    num_devices = torch.cuda.device_count()
    print("Available GPUs:")
    for i in range(num_devices):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    try:
        chosen = int(input(f"Select GPU index (0..{num_devices-1}) or -1 for CPU: "))
    except ValueError:
        chosen = -1
    if 0 <= chosen < num_devices:
        device = torch.device(f"cuda:{chosen}")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        if "RTX" in gpu_name.upper():
            amp_input = input("Your GPU supports Tensor Cores. Enable mixed precision AMP? [y/n]: ").strip().lower()
            use_amp = (amp_input == "y")
        else:
            use_amp = False

print(f"Using device: {device}")
print(f"Mixed precision (AMP) enabled: {use_amp}")

#############################
# Logging Configuration
#############################
import logging
logging.basicConfig(
    filename="error_log.txt",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG if EXTENDED_DEBUG else logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

#############################
# Neural Network Model: ChessDQN
#############################
class ChessDQN(nn.Module):
    def __init__(self):
        """
        Initialize the ChessDQN network with two branches: board and move.
        Their outputs are concatenated and passed through FC layers to produce an evaluation.
        """
        super(ChessDQN, self).__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.Flatten()  # 8192 features
        )
        self.move_fc = nn.Sequential(
            nn.Linear(MOVE_SIZE, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(8192 + 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        board_input = x[:, :STATE_SIZE]
        move_input = x[:, STATE_SIZE:]
        board_input = board_input.view(-1, 12, 8, 8)
        board_features = self.board_conv(board_input)
        move_features = self.move_fc(move_input)
        combined = torch.cat((board_features, move_features), dim=1)
        output = self.combined_fc(combined)
        return output

#############################
# Board Encoding Functions
#############################
def board_to_tensor(board):
    """Encode board state into a flattened vector with 12 channels."""
    piece_to_channel = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        channel = piece_to_channel[piece.piece_type]
        if piece.color == chess.WHITE:
            arr[channel, row, col] = 1.0
        else:
            arr[channel + 6, row, col] = 1.0
    return arr.flatten()

#############################
# Multithreaded Minimax Search with Alpha-Beta Pruning
#############################
def minimax_recursive(board, depth, alpha, beta, maximizing, agent_white, agent_black, end_time):
    """Recursive minimax with alpha-beta pruning."""
    if time.time() > end_time or depth == 0 or board.is_game_over():
        return (agent_white.evaluate_board(board)
                if board.turn == chess.WHITE
                else agent_black.evaluate_board(board))
    if maximizing:
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            value = minimax_recursive(board, depth - 1, alpha, beta, False, agent_white, agent_black, end_time)
            board.pop()
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_value
    else:
        best_value = math.inf
        for move in board.legal_moves:
            board.push(move)
            value = minimax_recursive(board, depth - 1, alpha, beta, True, agent_white, agent_black, end_time)
            board.pop()
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value

def minimax_with_time(board, depth, alpha, beta, maximizing, agent_white, agent_black, end_time):
    """Top-level minimax search using multithreading."""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1, None
        elif result == "0-1":
            return -1, None
        else:
            return 0, None
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return 0, None
    best_move = None
    best_value = -math.inf if maximizing else math.inf
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_move = {}
        for move in legal_moves:
            board.push(move)
            board_copy = board.copy()
            future = executor.submit(minimax_recursive, board_copy, depth - 1, alpha, beta,
                                      not maximizing, agent_white, agent_black, end_time)
            future_to_move[future] = move
            board.pop()
        for future in concurrent.futures.as_completed(future_to_move):
            try:
                value = future.result()
            except Exception as exc:
                logging.error(f"Error during minimax search: {exc}")
                value = 0
            move = future_to_move[future]
            if maximizing and value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, best_value)
            elif not maximizing and value < best_value:
                best_value = value
                best_move = move
                beta = min(beta, best_value)
    return best_value, best_move

#############################
# MCTS Definitions
#############################
class MCTSNode:
    """A node in the Monte Carlo Tree Search (MCTS) tree."""
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = 1.0

    def is_leaf(self):
        return len(self.children) == 0

    def puct_score(self, c_param=MCTS_EXPLORATION_PARAM):
        q_value = self.total_value / self.visits if self.visits > 0 else 0
        u_value = c_param * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits) if self.parent else 0
        return q_value + u_value

def mcts_search(root_board, neural_agent, num_simulations=MCTS_SIMULATIONS):
    """Perform MCTS starting from the given board state."""
    root = MCTSNode(root_board.copy())
    legal_moves = list(root.board.legal_moves)
    for move in legal_moves:
        new_board = root.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=root, move=move)
        child.prior = 1.0 / len(legal_moves)
        root.children[move] = child

    for sim in range(num_simulations):
        node = root
        search_path = [node]
        while not node.is_leaf() and not node.board.is_game_over():
            node = max(node.children.values(), key=lambda n: n.puct_score())
            search_path.append(node)
        if not node.board.is_game_over():
            legal_moves = list(node.board.legal_moves)
            if legal_moves:
                for move in legal_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    child = MCTSNode(new_board, parent=node, move=move)
                    child.prior = 1.0 / len(legal_moves)
                    node.children[move] = child
            value = neural_agent.evaluate_board(node.board)
            leaf_value = value
        else:
            result = node.board.result()
            if result == "1-0":
                leaf_value = 1
            elif result == "0-1":
                leaf_value = -1
            else:
                leaf_value = 0
        for n in reversed(search_path):
            n.visits += 1
            n.total_value += leaf_value
            leaf_value = -leaf_value

    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

#############################
# StatsManager Class: Record Training Statistics
#############################
class StatsManager:
    def __init__(self, filename=STATS_FILE):
        self.filename = filename
        self.load_stats()

    def load_stats(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "rb") as f:
                    data = pickle.load(f)
                self.wins_white = data.get("wins_white", 0)
                self.wins_black = data.get("wins_black", 0)
                self.draws = data.get("draws", 0)
                self.total_games = data.get("total_games", 0)
                self.global_move_count = data.get("global_move_count", 0)
                self.accumulated_training_time = data.get("accumulated_training_time", 0)
                self.first_mover_stats = data.get("first_mover_stats", {
                    "white": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0},
                    "black": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0}
                })
                logging.debug("Stats loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading stats: {e}. Initializing new stats.")
                self.wins_white = self.wins_black = self.draws = self.total_games = self.global_move_count = self.accumulated_training_time = 0
                self.first_mover_stats = {"white": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0},
                                           "black": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0}}
        else:
            self.wins_white = self.wins_black = self.draws = self.total_games = self.global_move_count = self.accumulated_training_time = 0
            self.first_mover_stats = {"white": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0},
                                       "black": {"first_count": 0, "wins_when_first": 0, "losses_when_first": 0}}

    def save_stats(self):
        data = {
            "wins_white": self.wins_white,
            "wins_black": self.wins_black,
            "draws": self.draws,
            "total_games": self.total_games,
            "global_move_count": self.global_move_count,
            "accumulated_training_time": self.accumulated_training_time,
            "first_mover_stats": self.first_mover_stats
        }
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(data, f)
            logging.debug("Stats saved successfully.")
        except Exception as e:
            logging.error(f"Error saving stats: {e}")

    def record_result(self, result_str):
        """Update stats based on game result."""
        self.total_games += 1
        if result_str == "1-0":
            self.wins_white += 1
        elif result_str == "0-1":
            self.wins_black += 1
        else:
            self.draws += 1

stats_manager = StatsManager()

def update_training_time():
    global training_active_start_time
    if training_active_start_time is not None:
        elapsed = time.time() - training_active_start_time
        stats_manager.accumulated_training_time += elapsed
        training_active_start_time = time.time()
        logging.debug(f"Updated training time by {elapsed:.2f} seconds.")

def get_total_training_time():
    if training_active_start_time is not None:
        return stats_manager.accumulated_training_time + (time.time() - training_active_start_time)
    else:
        return stats_manager.accumulated_training_time

def print_ascii_stats(stats):
    os.system('cls' if os.name == 'nt' else 'clear')
    model_paths = [MODEL_SAVE_PATH_WHITE, MODEL_SAVE_PATH_BLACK, MASTER_MODEL_SAVE_PATH]
    largest_model_path = max(model_paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
    largest_model_size = os.path.getsize(largest_model_path) if os.path.exists(largest_model_path) else 0
    table_paths = [TABLE_SAVE_PATH_WHITE, TABLE_SAVE_PATH_BLACK, MASTER_TABLE_SAVE_PATH]
    largest_table_path = max(table_paths, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
    largest_table_size = os.path.getsize(largest_table_path) if os.path.exists(largest_table_path) else 0
    print("=" * 60)
    print("    ULTRA-INTELLIGENT SELF-AWARE CHESS AI TRAINER STATS    ")
    print("=" * 60)
    print(f" Games:         {stats.total_games}")
    print(f" White Wins:    {stats.wins_white}")
    print(f" Black Wins:    {stats.wins_black}")
    print(f" Draws:         {stats.draws}")
    print(f" Global Moves:  {stats.global_move_count}")
    master_model_size_str = format_file_size(os.path.getsize(MASTER_MODEL_SAVE_PATH)) if os.path.exists(MASTER_MODEL_SAVE_PATH) else "N/A"
    master_table_size_str = format_file_size(os.path.getsize(MASTER_TABLE_SAVE_PATH)) if os.path.exists(MASTER_TABLE_SAVE_PATH) else "N/A"
    print(f" Master Files:  {MASTER_MODEL_SAVE_PATH} ({master_model_size_str}), {MASTER_TABLE_SAVE_PATH} ({master_table_size_str})")
    print(f" Largest Model File: {largest_model_path} ({format_file_size(largest_model_size)})")
    print(f" Largest Table File: {largest_table_path} ({format_file_size(largest_table_size)})")
    total_training_time = get_total_training_time()
    print(f" Training Time: {format_duration(total_training_time)}")
    if stats.total_games > 0:
        avg_moves = stats.global_move_count / stats.total_games
        avg_game_time = stats.accumulated_training_time / stats.total_games
        games_per_hour = (stats.total_games / (stats.accumulated_training_time / 3600)
                          if stats.accumulated_training_time > 0 else 0)
    else:
        avg_moves = avg_game_time = games_per_hour = 0
    print(f" Avg Moves/Game: {avg_moves:.1f}")
    print(f" Avg Time/Game:  {avg_game_time:.1f} s")
    print(f" Games/Hour:     {games_per_hour:.2f}")
    print(f" Training on:    {get_training_mode()}")
    if device.type == "cpu":
        mem = psutil.virtual_memory()
        print(f" RAM Used: {format_file_size(mem.used)} / {format_file_size(mem.total)} ({mem.percent}% used)")
    else:
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(f" GPU RAM Allocated: {format_file_size(allocated)}, Reserved: {format_file_size(reserved)}")
    print("=" * 60)

#############################
# Functions to Merge and Recover Master Data
#############################
def merge_state_dicts(state_dict1, state_dict2):
    merged = {}
    all_keys = set(state_dict1.keys()).union(set(state_dict2.keys()))
    for key in all_keys:
        if key in state_dict1 and key in state_dict2:
            merged[key] = (state_dict1[key] + state_dict2[key]) / 2.0
        elif key in state_dict1:
            merged[key] = state_dict1[key]
        else:
            merged[key] = state_dict2[key]
    return merged

def merge_transposition_tables(table1, table2):
    merged = table1.copy()
    for key, val in table2.items():
        if key in merged:
            merged[key] = (merged[key] + val) / 2.0
        else:
            merged[key] = val
    return merged

def check_and_repair_model_files():
    paths = [MODEL_SAVE_PATH_WHITE, MODEL_SAVE_PATH_BLACK, MASTER_MODEL_SAVE_PATH]
    sizes = {p: os.path.getsize(p) if os.path.exists(p) else 0 for p in paths}
    largest_path = max(sizes, key=sizes.get)
    largest_size = sizes[largest_path]
    for p in paths:
        if p != largest_path and sizes[p] < largest_size:
            try:
                with open(largest_path, "rb") as fin:
                    data = fin.read()
                with open(p, "wb") as fout:
                    fout.write(data)
                logging.info(f"Repaired model file {p} using data from {largest_path}.")
            except Exception as e:
                logging.error(f"Error repairing model file {p} from {largest_path}: {e}")

def check_and_repair_table_files():
    paths = [TABLE_SAVE_PATH_WHITE, TABLE_SAVE_PATH_BLACK, MASTER_TABLE_SAVE_PATH]
    sizes = {p: os.path.getsize(p) if os.path.exists(p) else 0 for p in paths}
    largest_path = max(sizes, key=sizes.get)
    largest_size = sizes[largest_path]
    for p in paths:
        if p != largest_path and sizes[p] < largest_size:
            try:
                with open(largest_path, "rb") as fin:
                    data = fin.read()
                with open(p, "wb") as fout:
                    fout.write(data)
                logging.info(f"Repaired table file {p} using data from {largest_path}.")
            except Exception as e:
                logging.error(f"Error repairing table file {p} from {largest_path}: {e}")

def check_and_repair_all_files():
    check_and_repair_model_files()
    check_and_repair_table_files()

def update_master_in_memory(agent_white, agent_black):
    global MASTER_MODEL_RAM, MASTER_TABLE_RAM
    with master_lock:
        white_state = agent_white.policy_net.state_dict()
        black_state = agent_black.policy_net.state_dict()
        new_master_state = merge_state_dicts(white_state, black_state)
        if MASTER_MODEL_RAM is not None:
            MASTER_MODEL_RAM = merge_state_dicts(MASTER_MODEL_RAM, new_master_state)
        else:
            MASTER_MODEL_RAM = new_master_state
        with agent_white.table_lock:
            white_table = agent_white.transposition_table.copy()
        with agent_black.table_lock:
            black_table = agent_black.transposition_table.copy()
        new_master_table = merge_transposition_tables(white_table, black_table)
        if MASTER_TABLE_RAM:
            MASTER_TABLE_RAM = merge_transposition_tables(MASTER_TABLE_RAM, new_master_table)
        else:
            MASTER_TABLE_RAM = new_master_table
        logging.debug("Master model and table updated in memory.")

def update_master_with_agent(agent):
    global MASTER_MODEL_RAM, MASTER_TABLE_RAM
    with master_lock:
        current_state = agent.policy_net.state_dict()
        if MASTER_MODEL_RAM is not None:
            MASTER_MODEL_RAM = merge_state_dicts(MASTER_MODEL_RAM, current_state)
        else:
            MASTER_MODEL_RAM = current_state
        with agent.table_lock:
            current_table = agent.transposition_table.copy()
        if MASTER_TABLE_RAM:
            MASTER_TABLE_RAM = merge_transposition_tables(MASTER_TABLE_RAM, current_table)
        else:
            MASTER_TABLE_RAM = current_table
    flush_master_to_disk()
    load_master_into_agent(agent)
    logging.debug(f"Master updated with agent {agent.name}.")

def flush_master_to_disk():
    global MASTER_MODEL_RAM, MASTER_TABLE_RAM
    if MASTER_MODEL_RAM is not None:
        try:
            torch.save(MASTER_MODEL_RAM, MASTER_MODEL_SAVE_PATH)
        except Exception as e:
            logging.error(f"Error flushing master model to disk: {e}")
    try:
        with open(MASTER_TABLE_SAVE_PATH, "wb") as f:
            pickle.dump(MASTER_TABLE_RAM, f)
    except Exception as e:
        logging.error(f"Error flushing master table to disk: {e}")

def load_master_into_agent(agent):
    global MASTER_MODEL_RAM, MASTER_TABLE_RAM
    if MASTER_MODEL_RAM is not None:
        try:
            agent.policy_net.load_state_dict(MASTER_MODEL_RAM, strict=False)
        except Exception as e:
            logging.error(f"Error loading RAM master model for {agent.name}: {e}")
    else:
        if os.path.exists(MASTER_MODEL_SAVE_PATH):
            try:
                master_state = torch.load(MASTER_MODEL_SAVE_PATH, map_location=device)
                agent.policy_net.load_state_dict(master_state, strict=False)
            except Exception as e:
                logging.error(f"Error loading master model from disk for {agent.name}: {e}")
    if MASTER_TABLE_RAM:
        try:
            agent.transposition_table = MASTER_TABLE_RAM.copy()
        except Exception as e:
            logging.error(f"Error loading master table from RAM for {agent.name}: {e}")
    else:
        if os.path.exists(MASTER_TABLE_SAVE_PATH):
            try:
                with open(MASTER_TABLE_SAVE_PATH, "rb") as f:
                    agent.transposition_table = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading master transposition table from disk for {agent.name}: {e}")

def recover_master_model():
    global MASTER_MODEL_RAM
    white_model, white_size = None, 0
    black_model, black_size = None, 0
    if os.path.exists(MODEL_SAVE_PATH_WHITE):
        white_model, white_size = torch.load(MODEL_SAVE_PATH_WHITE, map_location=device), os.path.getsize(MODEL_SAVE_PATH_WHITE)
    if os.path.exists(MODEL_SAVE_PATH_BLACK):
        black_model, black_size = torch.load(MODEL_SAVE_PATH_BLACK, map_location=device), os.path.getsize(MODEL_SAVE_PATH_BLACK)
    if white_model is None and black_model is None:
        logging.error("Both white and black model files unavailable or corrupt. Creating new model.")
        new_model = ChessDQN().to(device).state_dict()
        MASTER_MODEL_RAM = new_model
        torch.save(MASTER_MODEL_RAM, MASTER_MODEL_SAVE_PATH)
    elif white_model is not None and black_model is not None:
        if white_size >= black_size:
            MASTER_MODEL_RAM = merge_state_dicts(white_model, black_model)
        else:
            MASTER_MODEL_RAM = merge_state_dicts(black_model, white_model)
        torch.save(MASTER_MODEL_RAM, MASTER_MODEL_SAVE_PATH)
    else:
        MASTER_MODEL_RAM = white_model if white_model is not None else black_model
        torch.save(MASTER_MODEL_RAM, MASTER_MODEL_SAVE_PATH)
    logging.info("Master model recovered successfully.")

def recover_master_table():
    global MASTER_TABLE_RAM
    white_table, white_size = None, 0
    black_table, black_size = None, 0
    if os.path.exists(TABLE_SAVE_PATH_WHITE):
        with open(TABLE_SAVE_PATH_WHITE, "rb") as f:
            white_table = pickle.load(f)
        white_size = os.path.getsize(TABLE_SAVE_PATH_WHITE)
    if os.path.exists(TABLE_SAVE_PATH_BLACK):
        with open(TABLE_SAVE_PATH_BLACK, "rb") as f:
            black_table = pickle.load(f)
        black_size = os.path.getsize(TABLE_SAVE_PATH_BLACK)
    if white_table is None and black_table is None:
        logging.error("Both white and black transposition table files unavailable or corrupt. Creating new empty table.")
        MASTER_TABLE_RAM = {}
        with open(MASTER_TABLE_SAVE_PATH, "wb") as f:
            pickle.dump(MASTER_TABLE_RAM, f)
    elif white_table is not None and black_table is not None:
        if white_size >= black_size:
            MASTER_TABLE_RAM = merge_transposition_tables(white_table, black_table)
        else:
            MASTER_TABLE_RAM = merge_transposition_tables(black_table, white_table)
        with open(MASTER_TABLE_SAVE_PATH, "wb") as f:
            pickle.dump(MASTER_TABLE_RAM, f)
    else:
        MASTER_TABLE_RAM = white_table if white_table is not None else black_table
        with open(MASTER_TABLE_SAVE_PATH, "wb") as f:
            pickle.dump(MASTER_TABLE_RAM, f)
    logging.info("Master transposition table recovered successfully.")

def initial_master_sync():
    global MASTER_MODEL_RAM, MASTER_TABLE_RAM
    if os.path.exists(MASTER_MODEL_SAVE_PATH):
        try:
            MASTER_MODEL_RAM = torch.load(MASTER_MODEL_SAVE_PATH, map_location=device)
        except Exception as e:
            logging.error(f"Initial sync: Error loading master model: {e}")
            try:
                os.remove(MASTER_MODEL_SAVE_PATH)
            except Exception as re:
                logging.error(f"Initial sync: Error removing corrupt master model file: {re}")
            recover_master_model()
    else:
        logging.error("Initial sync: No master model file found. Recovery attempted from agents.")
    if os.path.exists(MASTER_TABLE_SAVE_PATH):
        try:
            with open(MASTER_TABLE_SAVE_PATH, "rb") as f:
                MASTER_TABLE_RAM = pickle.load(f)
        except Exception as e:
            logging.error(f"Initial sync: Error loading master table: {e}")
            try:
                os.remove(MASTER_TABLE_SAVE_PATH)
            except Exception as re:
                logging.error(f"Initial sync: Error removing corrupt master table file: {re}")
            recover_master_table()
    else:
        logging.error("Initial sync: No master table file found. Recovery attempted from agents.")
    check_and_repair_all_files()

def finalize_engine():
    check_and_repair_all_files()
    flush_master_to_disk()
    stats_manager.save_stats()
    logging.info("Engine finalized; all data saved to disk.")

atexit.register(finalize_engine)

#############################
# Background Saver Thread Function
#############################
def background_saver(agent_white, agent_black, stats_manager):
    while True:
        time.sleep(SAVE_INTERVAL_SECONDS)
        try:
            agent_white.save_model()
            agent_black.save_model()
            agent_white.save_transposition_table()
            agent_black.save_transposition_table()
            stats_manager.save_stats()
            flush_master_to_disk()
            logging.debug("Background saver: Models and stats saved.")
        except Exception as e:
            logging.error(f"Background saver error: {e}")

#############################
# ChessAgent Class: Core Agent for Self-Play
#############################
class ChessAgent:
    def __init__(self, name, model_path, table_path):
        self.name = name
        self.model_path = model_path
        self.table_path = table_path
        self.policy_net = ChessDQN().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPS_START
        self.steps = 0
        self.clock = INITIAL_CLOCK
        self.game_memory = []
        if os.path.exists(self.model_path):
            if not os.path.getsize(self.model_path) >= 1024 * 1024:
                logging.error(f"{self.model_path} is too small; repairing from master if available.")
                if os.path.exists(MASTER_MODEL_SAVE_PATH):
                    master_state = torch.load(MASTER_MODEL_SAVE_PATH, map_location=device)
                    self.policy_net.load_state_dict(master_state)
                    torch.save(master_state, self.model_path)
                    logging.info(f"Repaired {self.name} model file from master.")
                else:
                    logging.error("Master model not available; initializing new model.")
            else:
                try:
                    self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device))
                    logging.debug(f"{self.name} model loaded from {self.model_path}.")
                except Exception as e:
                    logging.error(f"Error loading model from {self.model_path}: {e}. Initializing new model.")
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "rb") as f:
                    self.transposition_table = pickle.load(f)
                logging.debug(f"{self.name} transposition table loaded from {self.table_path}.")
            except Exception as e:
                logging.error(f"Error loading transposition table from {self.table_path}: {e}. Attempting recovery.")
                if MASTER_TABLE_RAM:
                    self.transposition_table = MASTER_TABLE_RAM.copy()
                else:
                    self.transposition_table = {}
                with open(self.table_path, "wb") as f:
                    pickle.dump(self.transposition_table, f)
        else:
            self.transposition_table = {}
        self.table_lock = threading.Lock()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        logging.debug(f"{self.name} model saved to {self.model_path}.")

    def save_transposition_table(self):
        with self.table_lock:
            table_copy = dict(self.transposition_table)
        try:
            with open(self.table_path, "wb") as f:
                pickle.dump(table_copy, f)
            logging.debug(f"{self.name} transposition table saved to {self.table_path}.")
        except Exception as e:
            logging.error(f"Error saving transposition table for {self.name}: {e}")

    def evaluate_board(self, board):
        fen = board.fen()
        with self.table_lock:
            if fen in self.transposition_table:
                return self.transposition_table[fen]
        state_vector = board_to_tensor(board)
        dummy_move = np.zeros(MOVE_SIZE, dtype=np.float32)
        inp = np.concatenate([state_vector, dummy_move])
        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    value = self.policy_net(inp_tensor).item()
            else:
                value = self.policy_net(inp_tensor).item()
        with self.table_lock:
            self.transposition_table[fen] = value
        return value

    def select_move(self, board, opponent_agent):
        is_white_turn = board.turn == chess.WHITE
        if (self.name == "white" and is_white_turn) or (self.name == "black" and not is_white_turn):
            state_vector = board_to_tensor(board)
            dummy_move = np.zeros(MOVE_SIZE, dtype=np.float32)
            self.game_memory.append(np.concatenate([state_vector, dummy_move]))
        self.steps += 1
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(moves)
        else:
            if USE_MCTS:
                logging.debug(f"{self.name} using MCTS for move selection.")
                return mcts_search(board, self, num_simulations=MCTS_SIMULATIONS)
            else:
                end_time = time.time() + MOVE_TIME_LIMIT
                _, best_move = minimax_with_time(board, depth=5, alpha=-math.inf, beta=math.inf,
                                                 maximizing=(board.turn == chess.WHITE),
                                                 agent_white=(self if board.turn == chess.WHITE else opponent_agent),
                                                 agent_black=(self if board.turn == chess.BLACK else opponent_agent),
                                                 end_time=end_time)
                return best_move

    def train_after_game(self, result):
        if not self.game_memory:
            return
        arr_states = np.array(self.game_memory, dtype=np.float32)
        arr_labels = np.array([result] * len(self.game_memory), dtype=np.float32).reshape(-1, 1)
        st_tensor = torch.tensor(arr_states, device=device)
        lb_tensor = torch.tensor(arr_labels, device=device)
        dataset_size = len(st_tensor)
        indices = np.arange(dataset_size)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        for _ in range(EPOCHS_PER_GAME):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx+BATCH_SIZE]
                batch_states = st_tensor[batch_indices]
                batch_labels = lb_tensor[batch_indices]
                self.optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.policy_net(batch_states)
                        loss = self.criterion(predictions, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    predictions = self.policy_net(batch_states)
                    loss = self.criterion(predictions, batch_labels)
                    loss.backward()
                    self.optimizer.step()
        self.game_memory = []
        logging.debug(f"{self.name} trained after game with result {result}.")

#############################
# Self-Aware Trainer Class: Monitors and Adjusts Training Dynamics
#############################
class SelfAwareTrainer:
    def __init__(self):
        self.agent_white = ChessAgent("white", MODEL_SAVE_PATH_WHITE, TABLE_SAVE_PATH_WHITE)
        self.agent_black = ChessAgent("black", MODEL_SAVE_PATH_BLACK, TABLE_SAVE_PATH_BLACK)
        load_master_into_agent(self.agent_white)
        load_master_into_agent(self.agent_black)
        self.saver_thread = threading.Thread(target=background_saver, args=(self.agent_white, self.agent_black, stats_manager), daemon=True)
        self.saver_thread.start()
        self.evaluation_history = []
        self.low_variance_count = 0  # Count consecutive low-variance scans
        global training_active_start_time
        if training_active_start_time is None:
            training_active_start_time = time.time()

    def run_self_play(self):
        game_counter = 0
        adjustment_interval = 10
        try:
            while True:
                board = chess.Board()
                if game_counter % 2 == 0:
                    first_mover = self.agent_white
                    second_mover = self.agent_black
                else:
                    first_mover = self.agent_black
                    second_mover = self.agent_white
                stats_manager.first_mover_stats[first_mover.name]["first_count"] += 1
                print(f"Game {game_counter + 1}: {first_mover.name.upper()} is the first mover (white).")
                while not board.is_game_over():
                    if board.turn == chess.WHITE:
                        current_agent = first_mover
                        opponent_agent = second_mover
                    else:
                        current_agent = second_mover
                        opponent_agent = first_mover
                    move_start = time.time()
                    move = current_agent.select_move(board, opponent_agent)
                    if move is None:
                        break
                    board.push(move)
                    elapsed_time = time.time() - move_start
                    first_mover.clock -= elapsed_time
                    second_mover.clock -= elapsed_time
                    stats_manager.global_move_count += 1
                result = board.result()
                if first_mover.name == "white":
                    if result == "1-0":
                        stats_manager.first_mover_stats["white"]["wins_when_first"] += 1
                    elif result == "0-1":
                        stats_manager.first_mover_stats["white"]["losses_when_first"] += 1
                elif first_mover.name == "black":
                    if result == "0-1":
                        stats_manager.first_mover_stats["black"]["wins_when_first"] += 1
                    elif result == "1-0":
                        stats_manager.first_mover_stats["black"]["losses_when_first"] += 1
                if result == "1-0":
                    first_mover.train_after_game(+1)
                    second_mover.train_after_game(-1)
                elif result == "0-1":
                    first_mover.train_after_game(-1)
                    second_mover.train_after_game(+1)
                else:
                    first_mover.train_after_game(0)
                    second_mover.train_after_game(0)
                update_training_time()
                stats_manager.record_result(result)
                update_master_in_memory(self.agent_white, self.agent_black)
                print_ascii_stats(stats_manager)
                outcome_str = ("Win" if ((first_mover.name == "white" and result=="1-0") or (first_mover.name=="black" and result=="0-1"))
                               else "Loss" if ((first_mover.name=="white" and result=="0-1") or (first_mover.name=="black" and result=="1-0"))
                               else "Draw")
                print(f"Game {game_counter + 1} result: First mover ({first_mover.name}) {outcome_str}.")
                game_counter += 1
                if game_counter % adjustment_interval == 0:
                    self.self_adjust_hyperparameters()
        except KeyboardInterrupt:
            update_training_time()
            print("\nTraining interrupted. Returning to main menu...")
            return

    def perform_evaluation_scan(self, num_boards=1000):
        boards = []
        for i in range(num_boards):
            board = chess.Board()
            nmoves = random.randint(10, 30)
            for _ in range(nmoves):
                if board.is_game_over():
                    break
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            boards.append(board)
        evaluations = []
        with torch.no_grad():
            for board in boards:
                state_vector = board_to_tensor(board)
                dummy_move = np.zeros(MOVE_SIZE, dtype=np.float32)
                inp = np.concatenate([state_vector, dummy_move])
                t_inp = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
                val = self.agent_white.policy_net(t_inp).item()
                evaluations.append(val)
        arr = np.array(evaluations, dtype=np.float32)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr))
        return mean_val, median_val, std_val

    def self_adjust_hyperparameters(self):
        global LEARNING_RATE  # Declare global at the beginning to avoid syntax errors
        mean_val, median_val, std_val = self.perform_evaluation_scan()
        self.evaluation_history.append((mean_val, median_val, std_val))
        print(f"Self-Awareness Scan: Mean={mean_val:.4f}, Median={median_val:.4f}, StdDev={std_val:.4f}")
        logging.info(f"Self-Awareness Scan: Mean={mean_val:.4f}, Median={median_val:.4f}, StdDev={std_val:.4f}")
        
        # Check if variance is below threshold
        if std_val < EVAL_STD_THRESHOLD:
            self.low_variance_count += 1
            print(f"Low variance count: {self.low_variance_count}")
        else:
            self.low_variance_count = 0

        if self.low_variance_count >= AGGRESSIVE_COUNT_THRESHOLD:
            aggressive_lr_factor = LR_ADJUST_FACTOR * 1.5
            aggressive_epsilon_factor = EPSILON_ADJUST_FACTOR * 1.5
            old_lr = LEARNING_RATE
            new_lr = old_lr * aggressive_lr_factor
            print(f"Aggressively adjusting Learning Rate: {old_lr} -> {new_lr}")
            logging.info(f"Aggressively adjusted Learning Rate: {old_lr} -> {new_lr}")
            LEARNING_RATE = new_lr
            for agent in [self.agent_white, self.agent_black]:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
            for agent in [self.agent_white, self.agent_black]:
                old_eps = agent.epsilon
                agent.epsilon = min(1.0, agent.epsilon * aggressive_epsilon_factor)
                print(f"Aggressively adjusted epsilon for {agent.name}: {old_eps:.4f} -> {agent.epsilon:.4f}")
                logging.info(f"Aggressively adjusted epsilon for {agent.name}: {old_eps:.4f} -> {agent.epsilon:.4f}")
            self.low_variance_count = 0
        elif std_val < EVAL_STD_THRESHOLD:
            old_lr = LEARNING_RATE
            new_lr = old_lr * LR_ADJUST_FACTOR
            print(f"Adjusting Learning Rate: {old_lr} -> {new_lr}")
            logging.info(f"Adjusted Learning Rate: {old_lr} -> {new_lr}")
            LEARNING_RATE = new_lr
            for agent in [self.agent_white, self.agent_black]:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
            for agent in [self.agent_white, self.agent_black]:
                old_eps = agent.epsilon
                agent.epsilon = min(1.0, agent.epsilon * EPSILON_ADJUST_FACTOR)
                print(f"Adjusted epsilon for {agent.name}: {old_eps:.4f} -> {agent.epsilon:.4f}")
                logging.info(f"Adjusted epsilon for {agent.name}: {old_eps:.4f} -> {agent.epsilon:.4f}")
        else:
            print("No hyperparameter adjustment needed based on evaluation scan.")
            logging.info("Evaluation variance acceptable; no adjustments made.")

#############################
# Main Execution: Start Self-Aware Training
#############################
if __name__ == "__main__":
    print("Starting Ultra-Intelligent Self-Aware Chess AI Trainer...")
    initial_master_sync()
    trainer = SelfAwareTrainer()
    trainer.run_self_play()
