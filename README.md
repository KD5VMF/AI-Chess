# Ultra-Powered Hybrid Chess AI Trainer

## Overview

The **Ultra-Powered Hybrid Chess AI Trainer** is a high-performance, self-improving chess AI designed for self-play training. It leverages a hybrid approach that combines:
- **Deep Reinforcement Learning** using a Deep Q-Network (DQN) for board evaluation.
- **Monte Carlo Tree Search (MCTS)** for move selection.
- **Minimax with Alpha-Beta Pruning** for deterministic evaluation.
- **Alternating First-Mover Training** to ensure balance in self-play.
- **File Recovery & Consistency Checks** to prevent corruption and data loss.
- **Performance Monitoring** displaying **CPU and GPU memory allocation**.

This system is **designed for high-end hardware** and supports **multi-threaded** execution with **optimized hyperparameters** for deep search and efficient training.

---

## Features

### 🤖 Hybrid AI Chess Engine
- Combines **Deep Learning (DQN), MCTS, and Minimax Search**.
- Uses **Multi-Threading and GPU Acceleration** for high-speed calculations.
- Implements **Tensor-based Board Evaluation** for real-time decision-making.

### 🔄 Alternating First-Mover Advantage
- Ensures that self-play games **alternate between white and black starting**.
- Records detailed statistics for **first-mover performance analysis**.

### 🛠️ Automatic File Repair & Recovery
- At **startup and shutdown**, the AI automatically checks model files:
  - Uses the **largest and most trained model** for data recovery.
  - Ensures transposition tables and neural networks remain **consistent**.

### 📊 Real-Time System Monitoring
- Displays **RAM usage (CPU/GPU)**.
- Tracks **average game duration, moves per game, and games per hour**.

### 🏎️ Optimized for High-Performance Machines
- **High batch sizes and deep searches** for AI training.
- **Multi-threaded parallel processing** to speed up evaluations.
- **Automatic model merging and reinforcement learning**.

---

## How It Works

### 🧠 Neural Network Architecture (ChessDQN)
The AI uses a **Deep Q-Network (DQN)** with convolutional layers to evaluate the board position:
1. **Board Representation** → 12x8x8 tensor (each piece type has its own channel).
2. **Move Encoding** → One-hot vector for move selection.
3. **Fusion Layer** → Combines board state and move evaluation.
4. **Final Output** → A **score predicting the best move**.

### 🔍 Monte Carlo Tree Search (MCTS)
1. **Selection** → The AI selects a promising move using the PUCT formula.
2. **Expansion** → Adds new moves to the tree.
3. **Evaluation** → Uses ChessDQN to assign a score to each move.
4. **Backpropagation** → Updates statistics up the search tree.

### 🎭 Minimax with Alpha-Beta Pruning
- **Minimax Search** recursively evaluates all legal moves.
- **Alpha-Beta Pruning** reduces search complexity, ignoring unpromising branches.
- **Multi-threading support** speeds up computations.

### 🔁 Alternating First-Mover Training
- In self-play, the **first player alternates every game**.
- Records **first-mover win/loss rates** to **track training fairness**.

### 🛠️ File Recovery System
- Automatically **detects corrupt or outdated files**.
- **Merges or repairs models** using the **largest available file**.

---

## Installation

### 🖥️ System Requirements
- **Windows 11 / Linux** (Tested on Windows 11 with Anaconda)
- **Python 3.8+** (Recommended: Anaconda Environment)
- **NVIDIA GPU** (for CUDA acceleration) or **High-Core CPU** (for parallel processing)

### 📦 Install Dependencies
Run the following commands:

```bash
# Create and activate a Python environment
conda create -n chessai python=3.8
conda activate chessai

# Install necessary libraries
pip install torch torchvision torchaudio chess matplotlib psutil
```

If using **GPU acceleration**, install PyTorch with CUDA:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### 🔄 Running the Program

Run the AI trainer:

```bash
python MasterAIChess.py
```

---

## Usage Instructions

### 📌 Main Menu Options
When you start the program, you can select from these modes:

1️⃣ **Self-Play Training (Faster)** - No board animation, pure AI training.  
2️⃣ **Self-Play Training (Slower)** - AI vs AI with a visual board.  
3️⃣ **Human vs AI (Graphical)** - Play against the AI with a GUI.  
4️⃣ **Toggle Debug Logging** - Enables extended debug information.  
❌ **Quit** - Exit the program and save all training data.

---

## Hyperparameter Configuration

| Parameter | Default Value | Effect |
|-----------|--------------|--------|
| `LEARNING_RATE` | `1e-3` | Higher = faster learning, but may overfit |
| `BATCH_SIZE` | `256` | Larger batch = more efficient but higher RAM usage |
| `EPOCHS_PER_GAME` | `5` | More epochs = better training per game |
| `EPS_START` | `1.0` | Initial exploration rate (random moves) |
| `EPS_DECAY` | `0.99999` | Decay rate for exploration |
| `USE_MCTS` | `True` | Enables Monte Carlo Tree Search |
| `MCTS_SIMULATIONS` | `2000` | More simulations = deeper search but slower moves |
| `MOVE_TIME_LIMIT` | `300.0` sec | Maximum time AI can spend per move |
| `SAVE_INTERVAL_SECONDS` | `60` | Frequency of model auto-saving |

These parameters can be adjusted in `MasterAIChess.py` to suit different hardware.

---

## File Structure

| File | Description |
|------|------------|
| `MasterAIChess.py` | Main program file |
| `white_dqn.pt` | Model file for white agent |
| `black_dqn.pt` | Model file for black agent |
| `master_dqn.pt` | Master model (merged) |
| `stats.pkl` | Training statistics file |
| `logs/error_log.txt` | Logs any errors that occur |

---

## System Performance Monitoring

**Displays real-time system stats:**
- **CPU Usage (%)** - Helps identify bottlenecks.
- **RAM Usage (CPU & GPU)** - Ensures memory is not overloaded.
- **Average Time per Move** - Shows AI decision efficiency.

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'psutil'`
Install missing dependency:
```bash
pip install psutil
```

### ❌ AI Makes Random Moves
If the AI is making **random moves**, **increase MCTS simulations** or **train for more games**.

### ❌ Program Crashes on Exit
Ensure all models are saved before quitting:
```bash
python MasterAIChess.py --save
```

---

## Contributing

You are welcome to contribute improvements! Fork the repository, make changes, and submit a **pull request**.

---

## License

This project is licensed under the **MIT License**.
