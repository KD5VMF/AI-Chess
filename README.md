# Re-attempting to create and save the README.md file

readme_content = """# Ultra-Powered Hybrid Chess AI Trainer

![Chess AI Trainer](https://via.placeholder.com/800x200?text=Ultra-Powered+Hybrid+Chess+AI+Trainer)  
*Mastering Self-Play Through Alternating First-Mover Advantage, Automated File Recovery, and Extensive Stats*

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Neural Network Architecture (ChessDQN)](#neural-network-architecture-chessdqn)
  - [Self-Play Training & Alternating First-Mover Strategy](#self-play-training--alternating-first-mover-strategy)
  - [Search Algorithms: MCTS and Minimax](#search-algorithms-mcts-and-minimax)
  - [File Recovery and Consistency Checks](#file-recovery-and-consistency-checks)
  - [System Resource Monitoring](#system-resource-monitoring)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters and Tuning](#hyperparameters-and-tuning)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Ultra-Powered Hybrid Chess AI Trainer is an advanced training engine that continuously improves a hybrid chess AI model. It leverages:
- **Deep Learning**: A custom deep convolutional network (ChessDQN) for board evaluation.
- **Monte Carlo Tree Search (MCTS)**: For stochastic exploration of move sequences.
- **Minimax Search with Alpha-Beta Pruning**: For deterministic evaluation of positions.

The engine employs an **alternating first-mover strategy** to mitigate the inherent advantage of playing white. In self-play, the agent that starts the game is tracked separately via detailed first-mover statistics. The system also features robust file recovery routines to ensure that models and transposition tables are consistent, even in the face of file corruption.

Additionally, the trainer continuously monitors system resource usage (CPU and GPU RAM) and displays comprehensive training statistics.

---

## Features

- **Hybrid AI Architecture**  
  Combines deep learning (via ChessDQN) with classical search techniques (MCTS, minimax).
  
- **Alternating First-Mover Self-Play**  
  Alternates the starting position in self-play to reduce first-move bias and provides detailed first-mover statistics.

- **Robust File Recovery**  
  On startup and shutdown, compares model and transposition table files (white, black, and master) and repairs any outdated or corrupt files by using the largest (most trained) version.

- **Extensive Statistics**  
  Displays overall game stats, average moves/game, training time, file sizes, and even real-time system resource usage.

- **High-Powered Hardware Optimizations**  
  Designed to fully leverage high-end systems with increased batch sizes, epochs, and deeper search iterations.

---

## How It Works

### Neural Network Architecture (ChessDQN)

The core evaluation engine is a deep convolutional network named **ChessDQN**:
- **Board Representation**: The board is encoded as a 12x8x8 tensor (each piece type in its own channel).
- **Move Encoding**: Moves are encoded as one-hot vectors.
- **Dual Branch Design**:  
  - **Convolutional Branch**: Processes board state through a series of convolutions and normalization layers.
  - **Fully-Connected Branch**: Processes the move vector.
- **Fusion Layer**: Concatenates the board and move features and outputs a single evaluation score.

### Self-Play Training & Alternating First-Mover Strategy

- **Self-Play**: Two agents (white and black) play against each other.  
- **Alternation**: Each game alternates the first mover, so one game the white agent starts (playing as white), the next the black agent starts.
- **First-Mover Stats**:  
  - Records how many times each agent started the game, wins and losses when they started.

### Search Algorithms: MCTS and Minimax

- **Monte Carlo Tree Search (MCTS)**:  
  - **Selection**: Uses PUCT to balance exploration and exploitation.
  - **Expansion**: Adds child nodes at leaf nodes.
  - **Evaluation**: Uses ChessDQN to evaluate new nodes.
  - **Backpropagation**: Updates statistics up the tree.
- **Minimax Search with Alpha-Beta Pruning**:  
  - Recursively evaluates moves with a fixed search depth.
  - Uses multi-threading to parallelize evaluation, reducing search time.

### File Recovery and Consistency Checks

- **Consistency Checks**: On startup and exit, the program compares file sizes for white, black, and master files.
- **Repair Mechanism**:  
  - The largest file is considered the best (most trained) and is used to repair smaller, outdated versions.
- **Automated Recovery**: If a file is corrupt or missing, the program recovers it from available data.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/UltraPoweredHybridChessAITrainer.git
   cd UltraPoweredHybridChessAITrainer
