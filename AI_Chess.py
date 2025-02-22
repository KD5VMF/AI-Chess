import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading  # For threading AI move computation
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import chess
import time
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button

# ---------------------------
# Global Unicode for Pieces
# ---------------------------
piece_unicode = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}
piece_unicode_gui = piece_unicode.copy()

# ---------------------------
# Global Famous Moves Database
# ---------------------------
FAMOUS_MOVES = {
    "Qf6": 100,
    "Bd6": 120,
    "Rxd4": 110,
    "Nf4": 90,
    "Qg2": 130,
    "Qh5": 70,
    "Bxh6": 110,
    "Rxf7": 140,
    "Bxf7+": 150,
    "Nxd5": 80,
    "Qd8+": 100,
    "Bc4": 75,
    "Qe6": 90,
    "Rf8#": 1000,
    "Bf7#": 1000,
    "Rxf8#": 1000,
    "Nf6+": 95,
    "Qd6": 80,
    "Bxe6": 100,
    "Qe7": 85,
    "Rd8": 80,
    "Qg4": 90,
    "Qh6": 95,
    "Rc8": 70,
    "Qd4": 85,
    "Rd6": 90,
    "Bf5": 95,
    "Rxd5": 100,
    "Nxe5": 110
}
# (Extend FAMOUS_MOVES as desired.)

# Global flag to enable MCTS.
USE_MCTS = True

# ---------------------------
# GPU Selection and Tensor Core Detection
# ---------------------------
num_devices = torch.cuda.device_count()
use_amp = False  # Flag for Automatic Mixed Precision (AMP)
if num_devices == 0:
    print("No CUDA devices found. Running on CPU.")
    device = torch.device("cpu")
else:
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
            amp_input = input("Your GPU is an RTX and supports Tensor Cores. Do you want to enable Tensor Core usage (mixed precision via AMP) for this run? [y/n]: ").strip().lower()
            use_amp = (amp_input == "y")
        else:
            use_amp = False

print(f"Using device: {device}")
print(f"Mixed precision (AMP) enabled: {use_amp}")

# ---------------------------
# Hyperparameters and File Paths
# ---------------------------
STATE_SIZE = 768         # 12 channels * 8x8 board = 768
MOVE_SIZE = 128          # 64 (from) + 64 (to)
INPUT_SIZE = STATE_SIZE + MOVE_SIZE

HIDDEN_SIZE = 512        # Number of neurons in hidden layers. (RTX 3060 12GB Use 12288 here. This will result in about 6Gb GDDR usage.)
MAX_SEARCH_DEPTH = 4     # Maximum search depth for minimax/MCTS
MOVE_TIME_LIMIT = 10.0   # Maximum seconds allowed for move computation
EPS_START = 1.0          # Initial exploration rate (random moves)
EPS_END = 0.05           # Minimum exploration rate
EPS_DECAY = 0.9999       # Decay per move

INITIAL_CLOCK = 300.0    # Informational clock (seconds)

# In no-GUI mode, we want to save to disk only once per minute:
SAVE_INTERVAL_SECONDS = 60

MODEL_SAVE_PATH_WHITE = "white_dqn.pt"
MODEL_SAVE_PATH_BLACK = "black_dqn.pt"
TABLE_SAVE_PATH_WHITE = "white_transposition.pkl"
TABLE_SAVE_PATH_BLACK = "black_transposition.pkl"

LEARNING_RATE = 1e-3     # Learning rate for the optimizer
BATCH_SIZE = 32          # Batch size for training
EPOCHS_PER_GAME = 3      # Number of training epochs per game

STATS_FILE = "stats.pkl" # File to store global statistics

# ------------------------------------------------
# StatsManager Class (with self-check)
# ------------------------------------------------
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
                print("Stats loaded successfully.")
            except Exception as e:
                print(f"Error loading stats: {e}. Initializing new stats.")
                self.wins_white = 0
                self.wins_black = 0
                self.draws = 0
                self.total_games = 0
                self.global_move_count = 0
        else:
            self.wins_white = 0
            self.wins_black = 0
            self.draws = 0
            self.total_games = 0
            self.global_move_count = 0

    def save_stats(self):
        data = {
            "wins_white": self.wins_white,
            "wins_black": self.wins_black,
            "draws": self.draws,
            "total_games": self.total_games,
            "global_move_count": self.global_move_count
        }
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(data, f)
            print("Stats saved successfully.")
        except Exception as e:
            print(f"Error saving stats: {e}")

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

# ------------------------------------------------
# Neural Network (ChessDQN) Class
# ------------------------------------------------
class ChessDQN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChessDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------
# Board and Move Encoding Functions
# ------------------------------------------------
def board_to_tensor(board):
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
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
    v = np.zeros(128, dtype=np.float32)
    v[move.from_square] = 1.0
    v[64 + move.to_square] = 1.0
    return v

# ------------------------------------------------
# MCTS Node and Search Functions
# ------------------------------------------------
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = {}  # move -> MCTSNode
        self.visits = 0
        self.total_value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c_param=1.4):
        # Return an unvisited child immediately to force exploration.
        for child in self.children.values():
            if child.visits == 0:
                return child
        # Otherwise, use the UCB formula.
        return max(self.children.values(), key=lambda node: (node.total_value / node.visits) +
                   c_param * math.sqrt(math.log(self.visits) / node.visits))

def mcts_search(root_board, neural_agent, num_simulations=100):
    root = MCTSNode(root_board.copy())
    for _ in range(num_simulations):
        node = root
        # Selection
        while not node.is_leaf() and not node.board.is_game_over():
            node = node.best_child()
        # Expansion
        if not node.board.is_game_over():
            for move in node.board.legal_moves:
                if move not in node.children:
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children[move] = MCTSNode(new_board, parent=node, move=move)
            if node.children:
                node = random.choice(list(node.children.values()))
        # Simulation / Evaluation
        value = neural_agent.evaluate_board(node.board)
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

# ------------------------------------------------
# ChessAgent Class (with Famous Moves & Optional MCTS)
# ------------------------------------------------
class ChessAgent:
    def __init__(self, name, model_path, table_path):
        self.name = name
        self.policy_net = ChessDQN(INPUT_SIZE, HIDDEN_SIZE).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPS_START
        self.steps = 0
        self.clock = INITIAL_CLOCK  # informational only
        self.model_path = model_path
        self.table_path = table_path
        self.transposition_table = {}
        self.game_memory = []
        if os.path.exists(self.model_path):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
                print(f"{self.name}: Loaded model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}. Initializing new model.")
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "rb") as f:
                    self.transposition_table = pickle.load(f)
                print(f"{self.name}: Loaded transposition table with {len(self.transposition_table)} entries.")
            except Exception as e:
                print(f"Error loading transposition table from {self.table_path}: {e}. Initializing empty table.")
                self.transposition_table = {}

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        print(f"{self.name}: Model saved to {self.model_path}")

    def save_transposition_table(self):
        with open(self.table_path, "wb") as f:
            pickle.dump(self.transposition_table, f)
        print(f"{self.name}: Transposition table saved with {len(self.transposition_table)} entries.")

    def evaluate_board(self, board):
        fen = board.fen()
        if fen in self.transposition_table:
            return self.transposition_table[fen]
        st = board_to_tensor(board)
        dm = np.zeros(MOVE_SIZE, dtype=np.float32)
        inp = np.concatenate([st, dm])
        inp_t = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    val = self.policy_net(inp_t).item()
            else:
                val = self.policy_net(inp_t).item()
        self.transposition_table[fen] = val
        return val

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
            st = board_to_tensor(board)
            dm = np.zeros(MOVE_SIZE, dtype=np.float32)
            self.game_memory.append(np.concatenate([st, dm]))
        self.steps += 1
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(moves)
        else:
            if USE_MCTS:
                return mcts_search(board, self, num_simulations=100)
            else:
                best_move = None
                best_score = -math.inf
                for move in moves:
                    score = self.evaluate_candidate_move(board, move)
                    if score > best_score:
                        best_score = score
                        best_move = move
                return best_move

    def iterative_deepening(self, board, opponent_agent):
        end_time = time.time() + MOVE_TIME_LIMIT
        best_move = None
        depth = 1
        while depth <= MAX_SEARCH_DEPTH and time.time() < end_time:
            # Note: minimax_with_time is assumed defined elsewhere.
            val, mv = minimax_with_time(
                board,
                depth,
                -math.inf,
                math.inf,
                board.turn == chess.WHITE,
                agent_white=(self if board.turn == chess.WHITE else opponent_agent),
                agent_black=(self if board.turn == chess.BLACK else opponent_agent),
                end_time=end_time
            )
            if mv is not None:
                best_move = mv
            depth += 1
        return best_move

    def train_after_game(self, result):
        if not self.game_memory:
            return
        arr_states = np.array(self.game_memory, dtype=np.float32)
        arr_labels = np.array([result] * len(self.game_memory), dtype=np.float32).reshape(-1, 1)
        st_t = torch.tensor(arr_states, device=device)
        lb_t = torch.tensor(arr_labels, device=device)
        ds = len(st_t)
        idxs = np.arange(ds)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        for _ in range(EPOCHS_PER_GAME):
            np.random.shuffle(idxs)
            for start_idx in range(0, ds, BATCH_SIZE):
                b_idx = idxs[start_idx:start_idx+BATCH_SIZE]
                batch_states = st_t[b_idx]
                batch_labels = lb_t[b_idx]
                self.optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.policy_net(batch_states)
                        loss = self.criterion(preds, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    preds = self.policy_net(batch_states)
                    loss = self.criterion(preds, batch_labels)
                    loss.backward()
                    self.optimizer.step()
        self.game_memory = []

# ------------------------------------------------
# Background Saver Thread for No-GUI Mode
# ------------------------------------------------
def background_saver(agent_white, agent_black, stats_manager):
    while True:
        time.sleep(SAVE_INTERVAL_SECONDS)
        try:
            agent_white.save_model()
            agent_black.save_model()
            agent_white.save_transposition_table()
            agent_black.save_transposition_table()
            stats_manager.save_stats()
            print("Background saver: Saved models, transposition tables, and stats.")
        except Exception as e:
            print(f"Background saver error: {e}")

# ------------------------------------------------
# Self-Play Training (Faster) Mode (No Visual)
# ------------------------------------------------
def self_play_training_faster():
    agent_white = ChessAgent("white", MODEL_SAVE_PATH_WHITE, TABLE_SAVE_PATH_WHITE)
    agent_black = ChessAgent("black", MODEL_SAVE_PATH_BLACK, TABLE_SAVE_PATH_BLACK)
    # Start the background saver thread (daemon mode so it doesn't block program exit)
    saver_thread = threading.Thread(target=background_saver, args=(agent_white, agent_black, stats_manager), daemon=True)
    saver_thread.start()
    while True:
        try:
            board = chess.Board()
            agent_white.clock = INITIAL_CLOCK
            agent_black.clock = INITIAL_CLOCK
            while not board.is_game_over():
                move_start = time.time()
                if board.turn == chess.WHITE:
                    mv = agent_white.select_move(board, agent_black)
                else:
                    mv = agent_black.select_move(board, agent_white)
                if mv is not None:
                    board.push(mv)
                elapsed = time.time() - move_start
                agent_white.clock -= elapsed
                agent_black.clock -= elapsed
                stats_manager.global_move_count += 1

            if board.is_game_over():
                res = board.result()
                stats_manager.record_result(res)
                if res == "1-0":
                    agent_white.train_after_game(+1)
                    agent_black.train_after_game(-1)
                elif res == "0-1":
                    agent_white.train_after_game(-1)
                    agent_black.train_after_game(+1)
                else:
                    agent_white.train_after_game(0)
                    agent_black.train_after_game(0)

            # Instead of saving at game end, we rely on the background saver.
            print(f"Faster Training Stats: {stats_manager}")
        except KeyboardInterrupt:
            print("Stopping faster self-play training...")
            break

# ------------------------------------------------
# Self-Play Training (Slower) with Visual Mode (AI vs AI)
# (GUI modes remain unchanged)
# ------------------------------------------------
class SelfPlayGUI:
    def __init__(self):
        self.agent_white = ChessAgent("white", MODEL_SAVE_PATH_WHITE, TABLE_SAVE_PATH_WHITE)
        self.agent_black = ChessAgent("black", MODEL_SAVE_PATH_BLACK, TABLE_SAVE_PATH_BLACK)
        self.board = chess.Board()
        self.move_counter = 0
        self.current_game_start_time = time.time()
        self.fig = plt.figure(figsize=(8,6))
        self.ax_board = self.fig.add_axes([0.3, 0.05, 0.65, 0.9])
        self.ax_info = self.fig.add_axes([0.02, 0.05, 0.25, 0.9])
        self.ax_info.axis('off')
        self.ax_reset = self.fig.add_axes([0.3, 0.96, 0.1, 0.04])
        self.ax_stop = self.fig.add_axes([0.45, 0.96, 0.1, 0.04])
        self.ax_save = self.fig.add_axes([0.6, 0.96, 0.1, 0.04])
        self.btn_reset = Button(self.ax_reset, "Reset")
        self.btn_stop = Button(self.ax_stop, "Stop")
        self.btn_save = Button(self.ax_save, "Save")
        self.btn_reset.on_clicked(self.reset_callback)
        self.btn_stop.on_clicked(self.stop_callback)
        self.btn_save.on_clicked(self.save_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.manager.set_window_title("AI vs AI (Self-Play)")
        self.draw_board()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000, blit=False, cache_frame_data=False)
        plt.show()

    def reset_callback(self, event):
        self.board = chess.Board()
        self.move_counter = 0
        self.current_game_start_time = time.time()
        self.draw_board()
        print("Board reset.")

    def stop_callback(self, event):
        self.save_callback(event)
        plt.close(self.fig)
        print("Game stopped.")

    def save_callback(self, event):
        self.agent_white.save_model()
        self.agent_black.save_model()
        self.agent_white.save_transposition_table()
        self.agent_black.save_transposition_table()
        stats_manager.save_stats()
        print("Game and model saved.")

    def on_key_press(self, event):
        if event.key.lower() == "ctrl+q":
            print("CTRL+Q pressed. Saving and quitting...")
            self.save_callback(event)
            plt.close(self.fig)

    def update(self, frame):
        if not self.board.is_game_over():
            move_start = time.time()
            if self.board.turn == chess.WHITE:
                mv = self.agent_white.select_move(self.board, self.agent_black)
            else:
                mv = self.agent_black.select_move(self.board, self.agent_white)
            if mv is not None:
                self.board.push(mv)
                self.move_counter += 1
            stats_manager.global_move_count += 1
        else:
            stats_manager.total_games += 1
            res = self.board.result()
            stats_manager.record_result(res)
            if res == "1-0":
                self.agent_white.train_after_game(+1)
                self.agent_black.train_after_game(-1)
            elif res == "0-1":
                self.agent_white.train_after_game(-1)
                self.agent_black.train_after_game(+1)
            else:
                self.agent_white.train_after_game(0)
                self.agent_black.train_after_game(0)
            self.board = chess.Board()
            self.move_counter = 0
            self.current_game_start_time = time.time()
            self.agent_white.save_model()
            self.agent_black.save_model()
            self.agent_white.save_transposition_table()
            self.agent_black.save_transposition_table()
            stats_manager.save_stats()
            print(f"Self-Play Stats: {stats_manager}")
        self.draw_board()
        return []

    def draw_board(self):
        self.ax_board.clear()
        light_sq = "#F0D9B5"
        dark_sq = "#B58863"
        for r in range(8):
            for f in range(8):
                c = light_sq if (r+f) % 2 == 0 else dark_sq
                rect = plt.Rectangle((f, r), 1, 1, facecolor=c)
                self.ax_board.add_patch(rect)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                sym = piece_unicode[piece.symbol()]
                self.ax_board.text(f+0.5, r+0.5, sym, fontsize=32, ha='center', va='center')
        self.ax_board.set_xlim(0,8)
        self.ax_board.set_ylim(0,8)
        self.ax_board.set_xticks([])
        self.ax_board.set_yticks([])
        self.ax_board.set_aspect('equal')
        game_time = time.time() - self.current_game_start_time
        precision_mode = "Tensor Cores (AMP)" if use_amp else "CUDA FP32"
        info = (f"Mode: AI vs AI (Self-Play)\n"
                f"Total Games: {stats_manager.total_games}\n"
                f"Global Moves: {stats_manager.global_move_count}\n"
                f"Current Game Moves: {self.move_counter}\n"
                f"Game Duration: {game_time:.1f}s\n"
                f"White Wins: {stats_manager.wins_white}\n"
                f"Black Wins: {stats_manager.wins_black}\n"
                f"Draws: {stats_manager.draws}\n"
                f"Precision Mode: {precision_mode}")
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0, 0.5, info, transform=self.ax_info.transAxes, va='center', ha='left', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))
        self.fig.canvas.draw_idle()

# ------------------------------------------------
# Human vs AI GUI (with Control Buttons, CTRL+Q, Detailed Stats)
# ------------------------------------------------
class GUIChessAgent:
    def __init__(self, ai_is_white):
        self.ai_is_white = ai_is_white
        self.model_path = MODEL_SAVE_PATH_WHITE if ai_is_white else MODEL_SAVE_PATH_BLACK
        self.table_path = TABLE_SAVE_PATH_WHITE if ai_is_white else TABLE_SAVE_PATH_BLACK
        self.policy_net = ChessDQN(INPUT_SIZE, HIDDEN_SIZE).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.transposition_table = {}
        self.game_memory = []
        if os.path.exists(self.model_path):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
                print(f"Human-vs-AI: Loaded AI model from {self.model_path}")
            except Exception as e:
                print(f"Error loading AI model from {self.model_path}: {e}. Initializing new model.")
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "rb") as f:
                    self.transposition_table = pickle.load(f)
                print(f"Human-vs-AI: Loaded transposition from {self.table_path}")
            except Exception as e:
                print(f"Error loading transposition table from {self.table_path}: {e}. Using empty table.")
                self.transposition_table = {}

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        print(f"AI model saved to {self.model_path}")

    def save_table(self):
        with open(self.table_path, "wb") as f:
            pickle.dump(self.transposition_table, f)
        print(f"AI table saved to {self.table_path}")

    def evaluate_board(self, board):
        fen = board.fen()
        if fen in self.transposition_table:
            return self.transposition_table[fen]
        st = board_to_tensor(board)
        dm = np.zeros(MOVE_SIZE, dtype=np.float32)
        inp = np.concatenate([st, dm])
        inp_t = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    val = self.policy_net(inp_t).item()
            else:
                val = self.policy_net(inp_t).item()
        self.transposition_table[fen] = val
        return val

    def select_move(self, board):
        if board.turn == self.ai_is_white:
            s = board_to_tensor(board)
            d = np.zeros(MOVE_SIZE, dtype=np.float32)
            self.game_memory.append(np.concatenate([s, d]))
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(moves)
        else:
            return self.iterative_deepening(board)

    def iterative_deepening(self, board):
        end_time = time.time() + MOVE_TIME_LIMIT
        best_move = None
        depth = 1
        while depth <= MAX_SEARCH_DEPTH and time.time() < end_time:
            val, mv = minimax_with_time(
                board,
                depth,
                -math.inf,
                math.inf,
                board.turn == chess.WHITE,
                agent_white=self,
                agent_black=self,
                end_time=end_time
            )
            if mv is not None:
                best_move = mv
            depth += 1
        return best_move

    def train_after_game(self, result):
        if not self.game_memory:
            return
        s_np = np.array(self.game_memory, dtype=np.float32)
        l_np = np.array([result] * len(self.game_memory), dtype=np.float32).reshape(-1, 1)
        s_t = torch.tensor(s_np, device=device)
        l_t = torch.tensor(l_np, device=device)
        ds = len(s_t)
        idxs = np.arange(ds)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        for _ in range(EPOCHS_PER_GAME):
            np.random.shuffle(idxs)
            for start_idx in range(0, ds, BATCH_SIZE):
                b_idx = idxs[start_idx:start_idx+BATCH_SIZE]
                batch_states = s_t[b_idx]
                batch_labels = l_t[b_idx]
                self.optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.policy_net(batch_states)
                        loss = self.criterion(preds, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    preds = self.policy_net(batch_states)
                    loss = self.criterion(preds, batch_labels)
                    loss.backward()
                    self.optimizer.step()
        self.game_memory = []

class HumanVsAIGUI:
    def __init__(self, human_is_white=True):
        self.human_is_white = human_is_white
        self.ai_agent = GUIChessAgent(not human_is_white)
        self.board = chess.Board()
        self.human_clock = INITIAL_CLOCK
        self.ai_clock = INITIAL_CLOCK
        self.last_click_time = time.time()
        self.selected_square = None
        self.status_message = "Your move" if self.board.turn == self.human_is_white else "AI is thinking..."
        self.move_counter = 0
        self.fig = plt.figure(figsize=(8,6))
        self.ax_board = self.fig.add_axes([0.3, 0.05, 0.65, 0.9])
        self.ax_info = self.fig.add_axes([0.02, 0.05, 0.25, 0.9])
        self.ax_info.axis('off')
        self.ax_reset = self.fig.add_axes([0.3, 0.96, 0.1, 0.04])
        self.ax_stop = self.fig.add_axes([0.45, 0.96, 0.1, 0.04])
        self.ax_save = self.fig.add_axes([0.6, 0.96, 0.1, 0.04])
        self.btn_reset = Button(self.ax_reset, "Reset")
        self.btn_stop = Button(self.ax_stop, "Stop")
        self.btn_save = Button(self.ax_save, "Save")
        self.btn_reset.on_clicked(self.reset_callback)
        self.btn_stop.on_clicked(self.stop_callback)
        self.btn_save.on_clicked(self.save_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.manager.set_window_title("Human vs AI (Graphical)")
        self.draw_board()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def reset_callback(self, event):
        self.board = chess.Board()
        self.move_counter = 0
        self.status_message = "Your move" if self.board.turn == self.human_is_white else "AI is thinking..."
        self.draw_board()
        print("Board reset.")

    def stop_callback(self, event):
        self.save_callback(event)
        plt.close(self.fig)
        print("Game stopped.")

    def save_callback(self, event):
        self.ai_agent.save_model()
        self.ai_agent.save_table()
        stats_manager.save_stats()
        print("Game and model saved.")

    def on_key_press(self, event):
        if event.key.lower() == "ctrl+q":
            print("CTRL+Q pressed. Saving and quitting...")
            self.save_callback(event)
            plt.close(self.fig)

    def on_click(self, event):
        if self.board.is_game_over():
            print("Game Over. Close the window.")
            return
        if event.inaxes == self.ax_board:
            file = int(event.xdata) if event.xdata is not None else -1
            rank = int(event.ydata) if event.ydata is not None else -1
            if file < 0 or file > 7 or rank < 0 or rank > 7:
                return
            sq = rank * 8 + file
            if self.board.turn == self.human_is_white:
                self.handle_human_click(sq)
            else:
                self.status_message = "AI is thinking..."
                self.draw_board()

    def handle_human_click(self, sq):
        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.human_is_white:
                self.selected_square = sq
                print(f"Selected {chess.SQUARE_NAMES[sq]}")
            else:
                print("Not your piece.")
        else:
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                dest_rank = chess.square_rank(sq)
                if (piece.color == chess.WHITE and dest_rank == 7) or (piece.color == chess.BLACK and dest_rank == 0):
                    mv = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
                else:
                    mv = chess.Move(self.selected_square, sq)
            else:
                mv = chess.Move(self.selected_square, sq)
            if mv in self.board.legal_moves:
                user_time = time.time() - self.last_click_time
                self.human_clock -= user_time  # informational only
                self.last_click_time = time.time()
                self.board.push(mv)
                self.selected_square = None
                self.status_message = "AI is thinking..."
                self.move_counter += 1
                self.draw_board()
                if self.move_counter % 10 == 0:
                    response = input("You've played 10 moves. Continue? (y/n): ").strip().lower()
                    if response != 'y':
                        self.finish_game(ai_win=False)
                        return
                if self.board.is_game_over():
                    self.handle_game_over()
                    return
                self.ai_move()
            else:
                print("Illegal move.")
                self.selected_square = None

    def ai_move(self):
        self.status_message = "AI is thinking..."
        self.draw_board()
        def compute_ai_move():
            start = time.time()
            mv = self.ai_agent.select_move(self.board)
            spent = time.time() - start
            self.ai_clock -= spent  # informational only
            if mv is not None:
                self.board.push(mv)
                print(f"AI played: {mv.uci()}")
            else:
                print("No moves for AI. Possibly stalemate.")
            self.move_counter += 1
            self.status_message = "Your move"
            self.draw_board()
            if self.board.is_game_over():
                self.handle_game_over()
                return
            if self.move_counter % 10 == 0:
                response = input("You've played 10 moves. Continue? (y/n): ").strip().lower()
                if response != 'y':
                    self.finish_game(ai_win=False)
                    return
        t = threading.Thread(target=compute_ai_move)
        t.start()

    def handle_game_over(self):
        print("Game Over:", self.board.result())
        res = self.board.result()
        if res == "1-0":
            ai_win = (not self.human_is_white)
        elif res == "0-1":
            ai_win = (not self.human_is_white)
        else:
            ai_win = None
        self.finish_game(ai_win)

    def finish_game(self, ai_win):
        if ai_win is True:
            final = +1
        elif ai_win is False:
            final = -1
        else:
            final = 0
        stats_manager.total_games += 1
        if final == +1:
            if not self.human_is_white:
                stats_manager.wins_black += 1
            else:
                stats_manager.wins_white += 1
        elif final == -1:
            if not self.human_is_white:
                stats_manager.wins_white += 1
            else:
                stats_manager.wins_black += 1
        else:
            stats_manager.draws += 1
        self.ai_agent.train_after_game(final)
        self.ai_agent.save_model()
        self.ai_agent.save_table()
        stats_manager.save_stats()
        print(f"Human vs AI Stats: {stats_manager}")

    def draw_board(self):
        self.ax_board.clear()
        light_sq = "#F0D9B5"
        dark_sq = "#B58863"
        for r in range(8):
            for f in range(8):
                c = light_sq if (r+f) % 2 == 0 else dark_sq
                rect = plt.Rectangle((f, r), 1, 1, facecolor=c)
                self.ax_board.add_patch(rect)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                sym = piece_unicode_gui[piece.symbol()]
                self.ax_board.text(f+0.5, r+0.5, sym, fontsize=32, ha='center', va='center')
        self.ax_board.set_xlim(0,8)
        self.ax_board.set_ylim(0,8)
        self.ax_board.set_xticks([])
        self.ax_board.set_yticks([])
        self.ax_board.set_aspect('equal')
        turn_str = "White" if self.board.turn else "Black"
        move_status = "Your move" if self.board.turn == self.human_is_white else "AI is thinking..."
        precision_mode = "Tensor Cores (AMP)" if use_amp else "CUDA FP32"
        info = (f"Turn: {turn_str}\n"
                f"Status: {self.status_message}\n"
                f"Precision Mode: {precision_mode}\n"
                f"Human Clock: {self.human_clock:.1f}\n"
                f"AI Clock: {self.ai_clock:.1f}\n"
                f"Moves played: {self.move_counter}")
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0, 0.5, info, transform=self.ax_info.transAxes, va='center', ha='left', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))
        self.fig.canvas.draw_idle()

# ------------------------------------------------
# Main Menu
# ------------------------------------------------
def main():
    print("Welcome back! Current saved stats are:")
    print(stats_manager)
    print("\nSelect a mode:")
    print("[1] Self-play training (Faster) - no board animation")
    print("[2] Self-play training (Slower) - AI vs AI with visual")
    print("[3] Human vs AI (Graphical)")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == '1':
        self_play_training_faster()
    elif choice == '2':
        SelfPlayGUI()
    else:
        color_input = input("Play as White (w) or Black (b)? [w/b]: ").strip().lower()
        if color_input not in ['w', 'b']:
            color_input = 'w'
        HumanVsAIGUI(human_is_white=(color_input == 'w'))

if __name__ == "__main__":
    main()
