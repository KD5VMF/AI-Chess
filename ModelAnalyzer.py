#!/usr/bin/env python
"""
ModelAnalyzer.py

Enhanced version that:
  - Loads a trained master model (master_dqn.pt) and transposition table (master_transposition.pkl).
  - Generates a large list of random chess positions (or boards from a file).
  - Lets user pick a start index 'a' and end index 'b' so that the script only evaluates and plots boards from indices [a, b).
  - Saves the final analysis (including partial-scan results) in report.txt.

Usage:
  python ModelAnalyzer.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import chess
import matplotlib.pyplot as plt

###############################################################################
# Board Encoding Function
###############################################################################
def board_to_tensor(board):
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        channel = piece_to_channel[piece.piece_type]
        if piece.color == chess.WHITE:
            arr[channel, row, col] = 1.0
        else:
            arr[channel + 6, row, col] = 1.0
    return arr.flatten()

###############################################################################
# Model Architecture
###############################################################################
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
            nn.Flatten()  # 128 * 8 * 8 = 8192
        )
        self.move_fc = nn.Sequential(
            nn.Linear(128, 256),
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
        board_input = x[:, :768]
        move_input  = x[:, 768:]
        board_input = board_input.view(-1, 12, 8, 8)
        board_features = self.board_conv(board_input)
        move_features  = self.move_fc(move_input)
        combined       = torch.cat((board_features, move_features), dim=1)
        output         = self.combined_fc(combined)
        return output

###############################################################################
# Main Analysis Program
###############################################################################
def main():
    model_path = "master_dqn.pt"
    table_path = "master_transposition.pkl"

    # 1. Load model
    if not os.path.exists(model_path):
        print(f"Trained model file {model_path} not found.")
        return
    print(f"Loading trained model from {model_path}...")
    master_state = torch.load(model_path, map_location=torch.device("cpu"))
    model = ChessDQN()
    model.load_state_dict(master_state)
    model.eval()
    print("Model loaded successfully, set to evaluation mode.")

    # 2. Load transposition table (optional)
    transposition_table = {}
    if os.path.exists(table_path):
        print(f"Loading transposition table from {table_path}...")
        with open(table_path, "rb") as f:
            transposition_table = pickle.load(f)
        print(f"Transposition table loaded with {len(transposition_table)} entries.")
    else:
        print(f"Transposition table file {table_path} not found.")

    # 3. Ask user for how many random positions we generate
    try:
        total_positions = int(input("Generate how many random positions total? [default=10000]: ").strip())
    except ValueError:
        total_positions = 10000
    # 4. Generate them
    boards = []
    print(f"Generating {total_positions} random chess positions, please wait...")
    for i in range(total_positions):
        board = chess.Board()
        # random # moves
        nmoves = random.randint(10,30)
        for _ in range(nmoves):
            if board.is_game_over():
                break
            mvlist = list(board.legal_moves)
            if not mvlist:
                break
            mv = random.choice(mvlist)
            board.push(mv)
        boards.append(board)
    print("Finished generating random boards.")

    # 5. Let user pick a start and end index
    # We can show them the 'size' => total_positions
    print(f"Indices range from 0..{total_positions-1}")
    try:
        start_idx = int(input("Enter start index [0..N-1, default=0]: ").strip())
    except ValueError:
        start_idx = 0
    if start_idx<0 or start_idx>= total_positions:
        start_idx=0
    try:
        end_idx = int(input(f"Enter end index   [start..N, default={total_positions}]: ").strip())
    except ValueError:
        end_idx = total_positions
    if end_idx<= start_idx or end_idx> total_positions:
        end_idx= total_positions

    # So we only evaluate boards from start_idx to end_idx-1
    sub_boards = boards[start_idx:end_idx]
    print(f"Will evaluate {len(sub_boards)} boards from index {start_idx} to {end_idx-1} inclusive.")

    # 6. Evaluate each board, store results
    model_evals = []
    print("Evaluating boards...")
    with torch.no_grad():
        for i, board in enumerate(sub_boards):
            # encode
            svec = board_to_tensor(board)
            dummy= np.zeros(128, dtype=np.float32)
            inp  = np.concatenate([svec, dummy])
            t_inp= torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            val  = model(t_inp).item()
            model_evals.append(val)
            if (i+1)%1000==0:
                print(f"Evaluated {i+1}/{len(sub_boards)} sub-boards")

    # 7. Summarize and Plot
    import matplotlib.pyplot as plt
    arr = np.array(model_evals, dtype=np.float32)
    mean_val   = np.mean(arr)
    median_val = np.median(arr)
    std_val    = np.std(arr)
    min_val    = np.min(arr)
    max_val    = np.max(arr)

    # Plot histogram of model's evaluation
    plt.figure(figsize=(10,6))
    plt.hist(arr, bins=20, color="blue", alpha=0.7)
    plt.title(f"Distribution of Model Evaluations on Boards[{start_idx}:{end_idx}]")
    plt.xlabel("Evaluation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 8. Write short report
    with open("report.txt","w", encoding="utf-8") as f:
        f.write("MODEL ANALYSIS REPORT (No-Animation Version)\n")
        f.write("============================================\n\n")
        f.write(f"Subset of random boards analyzed: indices {start_idx} .. {end_idx-1}\n")
        f.write(f"Number of boards evaluated: {len(sub_boards)}\n")
        f.write(f"Mean evaluation:   {mean_val:.4f}\n")
        f.write(f"Median evaluation: {median_val:.4f}\n")
        f.write(f"Std. Deviation:    {std_val:.4f}\n")
        f.write(f"Min / Max:         {min_val:.4f} / {max_val:.4f}\n\n")
        f.write("NOTE:\nYou can share this entire report with ChatGPT for further discussion.\n")

    print("Analysis complete. Wrote partial-scan results to report.txt.")

if __name__=="__main__":
    main()
