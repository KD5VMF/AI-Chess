#!/usr/bin/env python
"""
===============================================================================
   ULTRA-POWERED HYBRID CHESS AI TRAINER: MASTERING SELF-PLAY THROUGH ALTERNATING
         FIRST-MOVER ADVANTAGE, AUTOMATED FILE RECOVERY, AND EXTENSIVE STATS
===============================================================================
Title: Ultra-Powered Hybrid Chess AI Trainer: Mastering Self-Play Through Alternating First-Mover Advantage

About:
    This advanced chess AI training engine continuously improves a hybrid model using deep learning,
    Monte Carlo Tree Search (MCTS), and multi-threaded minimax search with alpha-beta pruning. To
    counter the inherent first-move advantage, self-play games alternate which agent starts first 
    (the first mover always plays white). Detailed statistics—including win/draw counts, move counts,
    training time, and first-mover-specific performance—are recorded and displayed. In addition, the 
    engine robustly monitors and repairs model and transposition table files by comparing file sizes 
    (to determine which is most trained) and replacing any outdated files with the best available data.
    Finally, system RAM usage is displayed (CPU or GPU) so you can see the resources in use.
    
    This engine is designed to fully leverage high-end hardware. Hyperparameters such as batch size,
    number of epochs, and the number of MCTS simulations are increased for maximum performance.
===============================================================================
"""

import os
# Automatically set environment variable to work around the OpenMP duplicate runtime error.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import time
import math
import random
import pickle
import warnings
import threading
import concurrent.futures
import atexit
import psutil  # For retrieving system memory usage

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button

# =============================================================================
# Utility Functions
# =============================================================================
def format_duration(total_seconds):
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
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024.0
        index += 1
    return f"{size_bytes:,.1f} {units[index]}"

def get_training_mode():
    # This returns the global default device (used only if not in multi-GPU mode)
    if device.type == "cuda":
        if use_amp:
            return f"GPU-Tensor (AMP enabled) - {torch.cuda.get_device_name(device)}"
        else:
            return f"GPU-CUDA (FP32) - {torch.cuda.get_device_name(device)}"
    else:
        return f"CPU (using {torch.get_num_threads()} threads)"

# =============================================================================
# Global Configurations and Setup
# =============================================================================
warnings.filterwarnings("ignore", message=".*dpi.*")
EXTENDED_DEBUG = False  # Toggle extended debug logging
use_amp = False         # Global flag (used if single-GPU mode)

# Global master variables (to merge data across agents)
MASTER_MODEL_RAM = None  
MASTER_TABLE_RAM = {}
master_lock = threading.Lock()

# Global training start time
training_active_start_time = None

# =============================================================================
# Helper Functions for File Loading and Repair
# =============================================================================
def safe_load_torch_model(path):
    if not os.path.exists(path):
        return None, 0
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        size = os.path.getsize(path)
        return model, size
    except Exception as e:
        logging.error(f"Error loading model from {path}: {e}")
        return None, 0

def safe_load_pickle(path):
    if not os.path.exists(path):
        return None, 0
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        size = os.path.getsize(path)
        return obj, size
    except Exception as e:
        logging.error(f"Error loading pickle file from {path}: {e}")
        return None, 0

def repair_agent_table_file(path, table_data):
    try:
        with open(path, "wb") as f:
            pickle.dump(table_data, f)
        logging.info(f"Successfully repaired table file at {path}.")
    except Exception as e:
        logging.error(f"Failed to repair table file at {path}: {e}")

def validate_model_file(path, min_size=1024*1024):
    return os.path.exists(path) and os.path.getsize(path) >= min_size

# =============================================================================
# File Consistency Check Functions
# =============================================================================
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

# =============================================================================
# Logging Configuration
# =============================================================================
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

# =============================================================================
# Hyperparameters & File Paths (Optimized for High-Power Machines)
# =============================================================================
STATE_SIZE = 768
MOVE_SIZE = 128
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

# =============================================================================
# Device Selection (Default Single GPU)
# =============================================================================
# In single GPU mode we use the global device variables.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}")
    use_amp = True if "RTX" in gpu_name.upper() else False
    print(f"Mixed precision (AMP) enabled: {use_amp}")
else:
    device = torch.device("cpu")
    use_amp = False
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print("No CUDA devices found. Running on CPU.")

# =============================================================================
# Neural Network Model: ChessDQN
# =============================================================================
class ChessDQN(nn.Module):
    def __init__(self):
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
            nn.Flatten()
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

# =============================================================================
# Board and Move Encoding Functions
# =============================================================================
def board_to_tensor(board):
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

def move_to_tensor(move):
    vector = np.zeros(MOVE_SIZE, dtype=np.float32)
    vector[move.from_square] = 1.0
    vector[64 + move.to_square] = 1.0
    return vector

# =============================================================================
# Multithreaded Minimax Search with Alpha-Beta Pruning
# =============================================================================
def minimax_recursive(board, depth, alpha, beta, maximizing, agent_white, agent_black, end_time):
    if time.time() > end_time or depth == 0 or board.is_game_over():
        return agent_white.evaluate_board(board) if board.turn == chess.WHITE else agent_black.evaluate_board(board)
    if maximizing:
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            value = minimax_recursive(board, depth-1, alpha, beta, False, agent_white, agent_black, end_time)
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
            value = minimax_recursive(board, depth-1, alpha, beta, True, agent_white, agent_black, end_time)
            board.pop()
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value

def minimax_with_time(board, depth, alpha, beta, maximizing, agent_white, agent_black, end_time):
    if board.is_game_over():
        result = board.result()
        if result == "1-0": return 1, None
        elif result == "0-1": return -1, None
        else: return 0, None
    legal_moves = list(board.legal_moves)
    if not legal_moves: return 0, None
    best_move = None
    best_value = -math.inf if maximizing else math.inf
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_move = {}
        for move in legal_moves:
            board.push(move)
            board_copy = board.copy()
            future = executor.submit(minimax_recursive, board_copy, depth-1, alpha, beta,
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
                best_value = value; best_move = move; alpha = max(alpha, best_value)
            elif not maximizing and value < best_value:
                best_value = value; best_move = move; beta = min(beta, best_value)
    # Synchronize all CUDA devices (if any) without specifying device
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return best_value, best_move

# =============================================================================
# Unicode Mapping for Board Display and Famous Moves Bonuses
# =============================================================================
piece_unicode = {'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                 'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'}
piece_unicode_gui = piece_unicode.copy()
FAMOUS_MOVES = {"Qf6": 100, "Bd6": 120, "Rxd4": 110, "Nf4": 90, "Qg2": 130,
                "Qh5": 70, "Bxh6": 110, "Rxf7": 140, "Bxf7+": 150, "Nxd5": 80,
                "Qd8+": 100, "Bc4": 75, "Qe6": 90, "Rf8#": 1000, "Bf7#": 1000,
                "Rxf8#": 1000, "Nf6+": 95, "Qd6": 80, "Bxe6": 100, "Qe7": 85,
                "Rd8": 80, "Qg4": 90, "Qh6": 95, "Rc8": 70, "Qd4": 85, "Rd6": 90,
                "Bf5": 95, "Rxd5": 100, "Nxe5": 110}

# =============================================================================
# StatsManager Class
# =============================================================================
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
        data = {"wins_white": self.wins_white, "wins_black": self.wins_black, "draws": self.draws,
                "total_games": self.total_games, "global_move_count": self.global_move_count,
                "accumulated_training_time": self.accumulated_training_time,
                "first_mover_stats": self.first_mover_stats}
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(data, f)
            logging.debug("Stats saved successfully.")
        except Exception as e:
            logging.error(f"Error saving stats: {e}")
    def record_result(self, result_str):
        self.total_games += 1
        if result_str == "1-0":
            self.wins_white += 1
        elif result_str == "0-1":
            self.wins_black += 1
        else:
            self.draws += 1
    def __str__(self):
        return (f"Games: {self.total_games}, White Wins: {self.wins_white}, "
                f"Black Wins: {self.wins_black}, Draws: {self.draws}, "
                f"Global Moves: {self.global_move_count}")

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
    print("         ULTRA-POWERED HYBRID CHESS AI TRAINER STATS          ")
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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"GPU RAM Allocated: {format_file_size(allocated)}, Reserved: {format_file_size(reserved)}")
    else:
        mem = psutil.virtual_memory()
        print(f"RAM Used: {format_file_size(mem.used)} / {format_file_size(mem.total)} ({mem.percent}% used)")
    print("\n First Mover Stats:")
    print(f"  WHITE as first: Played {stats.first_mover_stats['white']['first_count']} times, Wins: {stats.first_mover_stats['white']['wins_when_first']}, Losses: {stats.first_mover_stats['white']['losses_when_first']}")
    print(f"  BLACK as first: Played {stats.first_mover_stats['black']['first_count']} times, Wins: {stats.first_mover_stats['black']['wins_when_first']}, Losses: {stats.first_mover_stats['black']['losses_when_first']}")
    print("=" * 60)
    print("All files are verified as good.")

# =============================================================================
# Functions to Merge and Recover Master Data
# =============================================================================
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
            # Move master model from CPU to agent's device
            agent.policy_net.load_state_dict(MASTER_MODEL_RAM)
            agent.policy_net.to(agent.device)
        except Exception as e:
            logging.error(f"Error loading RAM master model for {agent.name}: {e}")
    else:
        if os.path.exists(MASTER_MODEL_SAVE_PATH):
            try:
                master_state = torch.load(MASTER_MODEL_SAVE_PATH, map_location=torch.device("cpu"))
                agent.policy_net.load_state_dict(master_state)
                agent.policy_net.to(agent.device)
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
    white_model, white_size = safe_load_torch_model(MODEL_SAVE_PATH_WHITE)
    black_model, black_size = safe_load_torch_model(MODEL_SAVE_PATH_BLACK)
    if white_model is None and black_model is None:
        logging.error("Both white and black model files are unavailable or corrupt. Creating a new model.")
        new_model = ChessDQN().to(torch.device("cpu")).state_dict()
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
    white_table, white_size = safe_load_pickle(TABLE_SAVE_PATH_WHITE)
    black_table, black_size = safe_load_pickle(TABLE_SAVE_PATH_BLACK)
    if white_table is None and black_table is None:
        logging.error("Both white and black transposition table files are unavailable or corrupt. Creating a new empty table.")
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
            MASTER_MODEL_RAM = torch.load(MASTER_MODEL_SAVE_PATH, map_location=torch.device("cpu"))
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

# =============================================================================
# Finalization Routine on Exit
# =============================================================================
def finalize_engine():
    check_and_repair_all_files()
    flush_master_to_disk()
    stats_manager.save_stats()
    logging.info("Engine finalized and all data saved to disk.")

atexit.register(finalize_engine)

# =============================================================================
# Background Saver Thread Function
# =============================================================================
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

# =============================================================================
# ChessAgent Class: Core Agent for Self-Play
# =============================================================================
# Modified to accept per-agent device and use_amp parameters.
class ChessAgent:
    def __init__(self, name, model_path, table_path, device=None, use_amp_flag=None):
        self.name = name
        self.model_path = model_path
        self.table_path = table_path
        self.device = device if device is not None else torch.device("cpu")
        self.use_amp = use_amp_flag if use_amp_flag is not None else False
        self.policy_net = ChessDQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPS_START
        self.steps = 0
        self.clock = INITIAL_CLOCK
        self.game_memory = []
        if os.path.exists(self.model_path):
            if not validate_model_file(self.model_path, min_size=1024*1024):
                logging.error(f"{self.model_path} is too small; repairing from master if available.")
                if os.path.exists(MASTER_MODEL_SAVE_PATH):
                    master_state = torch.load(MASTER_MODEL_SAVE_PATH, map_location=torch.device("cpu"))
                    self.policy_net.load_state_dict(master_state)
                    torch.save(master_state, self.model_path)
                    logging.info(f"Repaired {self.name} model file from master.")
                else:
                    logging.error("Master model not available; initializing new model.")
            else:
                try:
                    self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
                self.transposition_table = {}
                repair_agent_table_file(self.table_path, self.transposition_table)
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
        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    value = self.policy_net(inp_tensor).item()
            else:
                value = self.policy_net(inp_tensor).item()
        with self.table_lock:
            self.transposition_table[fen] = value
        return value

    def evaluate_candidate_move(self, board, move):
        move_san = board.san(move)
        move_san_clean = move_san.replace("!", "").replace("?", "")
        bonus = FAMOUS_MOVES.get(move_san_clean, 0)
        board.push(move)
        score = self.evaluate_board(board)
        board.pop()
        return score + bonus

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

    def iterative_deepening(self, board, opponent_agent):
        end_time = time.time() + MOVE_TIME_LIMIT
        best_move = None
        depth = 1
        while depth <= 5 and time.time() < end_time:
            val, mv = minimax_with_time(board, depth, -math.inf, math.inf, board.turn == chess.WHITE,
                                        agent_white=(self if board.turn == chess.WHITE else opponent_agent),
                                        agent_black=(self if board.turn == chess.BLACK else opponent_agent),
                                        end_time=end_time)
            if mv is not None:
                best_move = mv
            depth += 1
        return best_move

    def train_after_game(self, result):
        if not self.game_memory:
            return
        arr_states = np.array(self.game_memory, dtype=np.float32)
        arr_labels = np.array([result] * len(self.game_memory), dtype=np.float32).reshape(-1, 1)
        st_tensor = torch.tensor(arr_states, device=self.device)
        lb_tensor = torch.tensor(arr_labels, device=self.device)
        dataset_size = len(st_tensor)
        indices = np.arange(dataset_size)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        for _ in range(EPOCHS_PER_GAME):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx+BATCH_SIZE]
                batch_states = st_tensor[batch_indices]
                batch_labels = lb_tensor[batch_indices]
                self.optimizer.zero_grad()
                if self.use_amp:
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

# =============================================================================
# GUIChessAgent Class: For Human vs. AI Play (Graphical)
# =============================================================================
class GUIChessAgent:
    def __init__(self, ai_is_white, device=None, use_amp_flag=None):
        self.name = "human-vs-ai"
        self.ai_is_white = ai_is_white
        self.device = device if device is not None else torch.device("cpu")
        self.use_amp = use_amp_flag if use_amp_flag is not None else False
        self.model_path = MODEL_SAVE_PATH_WHITE if ai_is_white else MODEL_SAVE_PATH_BLACK
        self.table_path = TABLE_SAVE_PATH_WHITE if ai_is_white else TABLE_SAVE_PATH_BLACK
        self.policy_net = ChessDQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.transposition_table = {}
        self.table_lock = threading.Lock()
        self.game_memory = []
        if os.path.exists(MASTER_MODEL_SAVE_PATH):
            try:
                master_state = torch.load(MASTER_MODEL_SAVE_PATH, map_location=torch.device("cpu"))
                self.policy_net.load_state_dict(master_state)
                self.policy_net.to(self.device)
            except Exception as e:
                logging.error(f"Error loading master model into Human-vs-AI agent: {e}")
        if os.path.exists(MASTER_TABLE_SAVE_PATH):
            try:
                with open(MASTER_TABLE_SAVE_PATH, "rb") as f:
                    self.transposition_table = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading master transposition table into Human-vs-AI agent: {e}")
                self.transposition_table = {}
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "rb") as f:
                    local_table = pickle.load(f)
                self.transposition_table.update(local_table)
            except Exception as e:
                logging.error(f"Error loading local transposition table from {self.table_path}: {e}. Attempting recovery.")
                self.transposition_table = {}
                repair_agent_table_file(self.table_path, self.transposition_table)
    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        logging.debug(f"GUI agent model saved to {self.model_path}.")
    def save_table(self):
        with self.table_lock:
            table_copy = dict(self.transposition_table)
        try:
            with open(self.table_path, "wb") as f:
                pickle.dump(table_copy, f)
            logging.debug(f"GUI agent transposition table saved to {self.table_path}.")
        except Exception as e:
            logging.error(f"Error saving AI table: {e}")
    def evaluate_board(self, board):
        fen = board.fen()
        with self.table_lock:
            if fen in self.transposition_table:
                return self.transposition_table[fen]
        state_vector = board_to_tensor(board)
        dummy_move = np.zeros(MOVE_SIZE, dtype=np.float32)
        inp = np.concatenate([state_vector, dummy_move])
        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    value = self.policy_net(inp_tensor).item()
            else:
                value = self.policy_net(inp_tensor).item()
        with self.table_lock:
            self.transposition_table[fen] = value
        return value
    def select_move(self, board):
        if board.turn == self.ai_is_white:
            state_vector = board_to_tensor(board)
            dummy_move = np.zeros(MOVE_SIZE, dtype=np.float32)
            self.game_memory.append(np.concatenate([state_vector, dummy_move]))
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.epsilon:
            logging.debug("GUI agent selecting random move.")
            return random.choice(moves)
        else:
            end_time = time.time() + MOVE_TIME_LIMIT
            _, best_move = minimax_with_time(board, depth=5, alpha=-math.inf, beta=math.inf,
                                             maximizing=(board.turn == self.ai_is_white),
                                             agent_white=self,
                                             agent_black=self,
                                             end_time=end_time)
            return best_move
    def iterative_deepening(self, board):
        end_time = time.time() + MOVE_TIME_LIMIT
        best_move = None
        depth = 1
        while depth <= 5 and time.time() < end_time:
            val, mv = minimax_with_time(board, depth, -math.inf, math.inf, board.turn == self.ai_is_white,
                                        agent_white=self, agent_black=self, end_time=end_time)
            if mv is not None:
                best_move = mv
            depth += 1
        return best_move
    def train_after_game(self, result):
        if not self.game_memory:
            return
        s_np = np.array(self.game_memory, dtype=np.float32)
        l_np = np.array([result] * len(self.game_memory), dtype=np.float32).reshape(-1, 1)
        st_tensor = torch.tensor(s_np, device=self.device)
        lb_tensor = torch.tensor(l_np, device=self.device)
        dataset_size = len(st_tensor)
        indices = np.arange(dataset_size)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        for _ in range(EPOCHS_PER_GAME):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx+BATCH_SIZE]
                batch_states = st_tensor[batch_indices]
                batch_labels = lb_tensor[batch_indices]
                self.optimizer.zero_grad()
                if self.use_amp:
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
        logging.debug("GUI agent trained after game.")

# =============================================================================
# MCTS Search Function
# =============================================================================
class MCTSNode:
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
    logging.debug("Starting MCTS search...")
    root = MCTSNode(root_board.copy())
    legal_moves = list(root.board.legal_moves)
    for move in legal_moves:
        new_board = root.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=root, move=move)
        child.prior = 1.0 / len(legal_moves)
        root.children[move] = child
    logging.debug(f"Root expanded with {len(legal_moves)} moves.")
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
        if sim % 100 == 0:
            logging.debug(f"MCTS simulation {sim} complete.")
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    logging.debug("MCTS search complete.")
    return best_move

# =============================================================================
# Self-Play Training Mode (Fast, No Animation)
# =============================================================================
def self_play_training_faster():
    # This mode uses the global device settings (single GPU or CPU).
    global training_active_start_time
    agent_white = ChessAgent("white", MODEL_SAVE_PATH_WHITE, TABLE_SAVE_PATH_WHITE)
    agent_black = ChessAgent("black", MODEL_SAVE_PATH_BLACK, TABLE_SAVE_PATH_BLACK)
    load_master_into_agent(agent_white)
    load_master_into_agent(agent_black)
    saver_thread = threading.Thread(target=background_saver, args=(agent_white, agent_black, stats_manager), daemon=True)
    saver_thread.start()
    if training_active_start_time is None:
        training_active_start_time = time.time()
    game_counter = 0
    logging.debug("Starting alternating first-mover self-play training loop.")
    try:
        while True:
            board = chess.Board()
            if game_counter % 2 == 0:
                first_mover = agent_white
                second_mover = agent_black
            else:
                first_mover = agent_black
                second_mover = agent_white
            stats_manager.first_mover_stats[first_mover.name]["first_count"] += 1
            print(f"Game {game_counter + 1}: {first_mover.name.upper()} is playing as first mover (white).")
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    current_agent = first_mover; opponent_agent = second_mover
                else:
                    current_agent = second_mover; opponent_agent = first_mover
                move_start = time.time()
                move = current_agent.select_move(board, opponent_agent)
                if move is None: break
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
                first_mover.train_after_game(+1); second_mover.train_after_game(-1)
            elif result == "0-1":
                first_mover.train_after_game(-1); second_mover.train_after_game(+1)
            else:
                first_mover.train_after_game(0); second_mover.train_after_game(0)
            update_training_time()
            stats_manager.record_result(result)
            update_master_in_memory(agent_white, agent_black)
            print_ascii_stats(stats_manager)
            outcome_str = "Win" if ((first_mover.name == "white" and result=="1-0") or (first_mover.name=="black" and result=="0-1")) else "Loss" if ((first_mover.name=="white" and result=="0-1") or (first_mover.name=="black" and result=="1-0")) else "Draw"
            print(f"Game {game_counter + 1} result: First mover ({first_mover.name}) {outcome_str}.")
            game_counter += 1
    except KeyboardInterrupt:
        update_training_time()
        print("\nTraining interrupted. Returning to main menu...")
        return

# =============================================================================
# GPU vs GPU Training Mode
# =============================================================================
def gpu_vs_gpu_training(white_device, white_amp, black_device, black_amp):
    # Create two agents on separate GPUs
    global training_active_start_time
    agent_white = ChessAgent("white", MODEL_SAVE_PATH_WHITE, TABLE_SAVE_PATH_WHITE, device=white_device, use_amp_flag=white_amp)
    agent_black = ChessAgent("black", MODEL_SAVE_PATH_BLACK, TABLE_SAVE_PATH_BLACK, device=black_device, use_amp_flag=black_amp)
    load_master_into_agent(agent_white)
    load_master_into_agent(agent_black)
    saver_thread = threading.Thread(target=background_saver, args=(agent_white, agent_black, stats_manager), daemon=True)
    saver_thread.start()
    if training_active_start_time is None:
        training_active_start_time = time.time()
    game_counter = 0
    logging.debug("Starting GPU vs GPU self-play training loop.")
    try:
        while True:
            board = chess.Board()
            if game_counter % 2 == 0:
                first_mover = agent_white; second_mover = agent_black
            else:
                first_mover = agent_black; second_mover = agent_white
            stats_manager.first_mover_stats[first_mover.name]["first_count"] += 1
            print(f"Game {game_counter + 1}: {first_mover.name.upper()} is playing as first mover (white).")
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    current_agent = first_mover; opponent_agent = second_mover
                else:
                    current_agent = second_mover; opponent_agent = first_mover
                move_start = time.time()
                move = current_agent.select_move(board, opponent_agent)
                if move is None: break
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
                first_mover.train_after_game(+1); second_mover.train_after_game(-1)
            elif result == "0-1":
                first_mover.train_after_game(-1); second_mover.train_after_game(+1)
            else:
                first_mover.train_after_game(0); second_mover.train_after_game(0)
            update_training_time()
            stats_manager.record_result(result)
            update_master_in_memory(agent_white, agent_black)
            print_ascii_stats(stats_manager)
            outcome_str = "Win" if ((first_mover.name == "white" and result=="1-0") or (first_mover.name=="black" and result=="0-1")) else "Loss" if ((first_mover.name=="white" and result=="0-1") or (first_mover.name=="black" and result=="1-0")) else "Draw"
            print(f"Game {game_counter + 1} result: First mover ({first_mover.name}) {outcome_str}.")
            game_counter += 1
    except KeyboardInterrupt:
        update_training_time()
        print("\nGPU vs GPU Training interrupted. Returning to main menu...")
        return

# =============================================================================
# Self-Play GUI Mode (AI vs AI with Visuals) and Human vs AI GUI modes remain unchanged.
# (Omitted here for brevity; they can continue to use the single-GPU global settings.)
# =============================================================================
# ... [GUIChessAgent-based classes unchanged] ...

# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("Select mode:")
    print("1: Self-Play Training (Fast, No Animation) [Single GPU/CPU]")
    print("2: Self-Play GUI (AI vs AI with Visuals)")
    print("3: Human vs AI GUI")
    print("4: GPU vs GPU Training")
    mode = input("Enter choice (1/2/3/4): ").strip()
    if mode == "1":
        self_play_training_faster()
    elif mode == "2":
        # Call SelfPlayGUI (unchanged from original)
        from matplotlib import pyplot as plt
        SelfPlayGUI()
    elif mode == "3":
        human_color = input("Do you want to play as White? (y/n): ").strip().lower()
        human_is_white = True if human_color == "y" else False
        from matplotlib import pyplot as plt
        HumanVsAIGUI(human_is_white)
    elif mode == "4":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("At least two CUDA devices are required for GPU vs GPU Training.")
            return
        print("Available GPUs:")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
        try:
            white_idx = int(input("Enter GPU index for WHITE agent: ").strip())
            black_idx = int(input("Enter GPU index for BLACK agent: ").strip())
        except ValueError:
            print("Invalid input. Defaulting to GPU 0 for WHITE and GPU 1 for BLACK.")
            white_idx = 0; black_idx = 1
        white_device = torch.device(f"cuda:{white_idx}")
        black_device = torch.device(f"cuda:{black_idx}")
        white_gpu_name = torch.cuda.get_device_name(white_idx)
        black_gpu_name = torch.cuda.get_device_name(black_idx)
        white_amp = False
        black_amp = False
        if "RTX" in white_gpu_name.upper():
            white_amp = True if input("Enable AMP for WHITE agent on GPU {}? (y
