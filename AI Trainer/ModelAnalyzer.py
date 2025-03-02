#!/usr/bin/env python
"""
Enhanced ModelAnalyzer.py

About:
    This script loads a trained master model (master_dqn.pt) and its transposition table
    (master_transposition.pkl), then generates a large number of random chess positions.
    It allows you to specify a subset of these positions to evaluate and performs an
    advanced statistical analysis on the model’s evaluations. The analysis includes
    calculating the mean, median, standard deviation, min, max, skewness, kurtosis,
    and selected percentiles, along with generating a histogram, box plot, and Q–Q plot.
    The results are saved to a text report (report.txt) and the plots are displayed.
    
Usage:
    python EnhancedModelAnalyzer.py
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
from scipy.stats import skew, kurtosis, probplot

#############################################
# Board Encoding Function
#############################################
def board_to_tensor(board):
    """
    Convert a chess board into a flattened tensor with 12 channels.
    White pieces occupy channels 0-5 and black pieces channels 6-11.
    """
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

#############################################
# Model Architecture Definition
#############################################
class ChessDQN(nn.Module):
    def __init__(self):
        super(ChessDQN, self).__init__()
        # Note: This architecture should match the one used during training.
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
            nn.Flatten()  # Output: 128 * 8 * 8 = 8192 features
        )
        # The move branch: note that the training code uses 128 as move size.
        self.move_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        # Combined branch:
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
        # x is expected to be a 1D vector with size = 768 (board encoding) + 128 (dummy move)
        board_input = x[:, :768]
        move_input = x[:, 768:]
        board_input = board_input.view(-1, 12, 8, 8)
        board_features = self.board_conv(board_input)
        move_features = self.move_fc(move_input)
        combined = torch.cat((board_features, move_features), dim=1)
        output = self.combined_fc(combined)
        return output

#############################################
# Main Analysis Function
#############################################
def main():
    model_path = "master_dqn.pt"
    table_path = "master_transposition.pkl"
    
    # 1. Load the trained master model
    if not os.path.exists(model_path):
        print(f"Trained model file {model_path} not found.")
        sys.exit(1)
    print(f"Loading trained model from {model_path}...")
    master_state = torch.load(model_path, map_location=torch.device("cpu"))
    model = ChessDQN()
    model.load_state_dict(master_state)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    
    # 2. Load transposition table (if available)
    transposition_table = {}
    if os.path.exists(table_path):
        print(f"Loading transposition table from {table_path}...")
        with open(table_path, "rb") as f:
            transposition_table = pickle.load(f)
        print(f"Transposition table loaded with {len(transposition_table)} entries.")
    else:
        print(f"Transposition table file {table_path} not found.")
    
    # 3. Ask user for number of random positions to generate
    try:
        total_positions = int(input("Generate how many random positions total? [default=10000]: ").strip())
    except ValueError:
        total_positions = 10000
    
    # 4. Generate random chess positions
    boards = []
    print(f"Generating {total_positions} random chess positions, please wait...")
    for i in range(total_positions):
        board = chess.Board()
        nmoves = random.randint(10, 30)
        for _ in range(nmoves):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))
        boards.append(board)
    print("Finished generating random boards.")
    
    # 5. Let user select a subset for analysis
    print(f"Indices range from 0 to {total_positions - 1}")
    try:
        start_idx = int(input("Enter start index [default=0]: ").strip())
    except ValueError:
        start_idx = 0
    if start_idx < 0 or start_idx >= total_positions:
        start_idx = 0
    try:
        end_idx = int(input(f"Enter end index [default={total_positions}]: ").strip())
    except ValueError:
        end_idx = total_positions
    if end_idx <= start_idx or end_idx > total_positions:
        end_idx = total_positions
    sub_boards = boards[start_idx:end_idx]
    print(f"Evaluating {len(sub_boards)} boards (indices {start_idx} to {end_idx - 1}).")
    
    # 6. Evaluate boards using the master model
    evaluations = []
    print("Evaluating boards...")
    with torch.no_grad():
        for i, board in enumerate(sub_boards):
            svec = board_to_tensor(board)
            # Create a dummy move vector (all zeros)
            dummy_move = np.zeros(128, dtype=np.float32)
            inp = np.concatenate([svec, dummy_move])
            t_inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            val = model(t_inp).item()
            evaluations.append(val)
            if (i + 1) % 1000 == 0:
                print(f"Evaluated {i + 1}/{len(sub_boards)} boards")
    evaluations = np.array(evaluations, dtype=np.float32)
    
    # 7. Compute advanced statistics
    mean_val = np.mean(evaluations)
    median_val = np.median(evaluations)
    std_val = np.std(evaluations)
    min_val = np.min(evaluations)
    max_val = np.max(evaluations)
    skew_val = skew(evaluations)
    kurt_val = kurtosis(evaluations)
    percentiles = np.percentile(evaluations, [5, 25, 50, 75, 95])
    
    # 8. Display the advanced numerical analysis
    print("\nAdvanced Numerical Analysis:")
    print("============================================")
    print(f"Number of boards evaluated: {len(sub_boards)}")
    print(f"Mean evaluation:   {mean_val:.4f}")
    print(f"Median evaluation: {median_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")
    print(f"Minimum evaluation: {min_val:.4f}")
    print(f"Maximum evaluation: {max_val:.4f}")
    print(f"Skewness:          {skew_val:.4f}")
    print(f"Kurtosis:          {kurt_val:.4f}")
    print(f"5th percentile:    {percentiles[0]:.4f}")
    print(f"25th percentile:   {percentiles[1]:.4f}")
    print(f"50th percentile:   {percentiles[2]:.4f}")
    print(f"75th percentile:   {percentiles[3]:.4f}")
    print(f"95th percentile:   {percentiles[4]:.4f}")
    print("============================================")
    
    # 9. Plot the data: Histogram, Box Plot, and Q–Q Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(evaluations, bins=30, color="blue", alpha=0.7)
    plt.title("Histogram of Model Evaluations")
    plt.xlabel("Evaluation Value")
    plt.ylabel("Frequency")
    
    plt.subplot(2, 2, 2)
    plt.boxplot(evaluations, vert=False)
    plt.title("Box Plot of Model Evaluations")
    plt.xlabel("Evaluation Value")
    
    plt.subplot(2, 2, 3)
    probplot(evaluations, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Normality Check)")
    
    plt.tight_layout()
    plt.show()
    
    # 10. Write a detailed report to file
    with open("enhanced_report.txt", "w", encoding="utf-8") as f:
        f.write("ENHANCED MODEL ANALYSIS REPORT\n")
        f.write("============================================\n")
        f.write(f"Subset of random boards analyzed: indices {start_idx} .. {end_idx - 1}\n")
        f.write(f"Number of boards evaluated: {len(sub_boards)}\n")
        f.write(f"Mean evaluation:   {mean_val:.4f}\n")
        f.write(f"Median evaluation: {median_val:.4f}\n")
        f.write(f"Standard Deviation: {std_val:.4f}\n")
        f.write(f"Minimum evaluation: {min_val:.4f}\n")
        f.write(f"Maximum evaluation: {max_val:.4f}\n")
        f.write(f"Skewness:          {skew_val:.4f}\n")
        f.write(f"Kurtosis:          {kurt_val:.4f}\n")
        f.write(f"Percentiles (5th, 25th, 50th, 75th, 95th): {percentiles}\n")
        f.write("============================================\n")
        f.write("NOTE:\nShare this report for further discussion.\n")
    
    print("Enhanced analysis complete. Detailed report written to 'enhanced_report.txt'.")

if __name__ == "__main__":
    main()
