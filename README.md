# Chess DQN Reinforcement Learning Agent with MCTS and GUI

A Python-based chess engine that leverages deep reinforcement learning combined with Monte Carlo Tree Search (MCTS) for move evaluation. The program supports multiple modes—including fast self-play training (without board animation), an AI-vs-AI self-play visual mode, and a human-vs-AI interactive GUI—all built using PyTorch, python‑chess, and Matplotlib.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture and Code Structure](#architecture-and-code-structure)
  - [Neural Network: ChessDQN](#neural-network-chessdqn)
  - [Board and Move Encoding](#board-and-move-encoding)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [Chess Agents](#chess-agents)
  - [Self-Play Training Modes](#self-play-training-modes)
  - [Graphical User Interfaces (GUI)](#graphical-user-interfaces-gui)
  - [Statistics and Persistence](#statistics-and-persistence)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration and Hyperparameters](#configuration-and-hyperparameters)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a chess engine that trains two agents (playing White and Black) via self-play. The agents learn to evaluate board positions using a deep neural network architecture (a simple feed-forward network) and improve through reinforcement learning. The engine also integrates MCTS to help with move selection and supports a bonus evaluation for a predefined set of “famous moves” in chess.

Users can choose from several operational modes:

1. **Self-Play Training (Faster Mode):** Runs AI vs AI games without board animation for rapid training.
2. **Self-Play Training (Slower Mode with Visuals):** Displays animated board states during AI vs AI self-play.
3. **Human vs AI (Graphical):** An interactive GUI for playing against the AI with board clicks and control buttons.

---

## Features

- **Deep Reinforcement Learning:** Uses a DQN-style architecture to evaluate board positions.
- **Monte Carlo Tree Search (MCTS):** Optionally uses MCTS for enhanced move exploration.
- **Famous Moves Database:** Provides bonus evaluation scores for well-known tactical moves.
- **Multiple Training Modes:**
  - **Faster Self-Play:** No GUI, ideal for rapid training.
  - **Slower Self-Play with Visuals:** Visualizes AI vs AI games using Matplotlib animations.
  - **Human vs AI:** Interactive GUI with board clicks and key controls.
- **Transposition Table:** Caches board evaluations to avoid redundant computation.
- **GPU Support:** Automatically detects CUDA-enabled devices and lets the user choose which GPU to use (or fallback to CPU).
- **Periodic Saving:** Models and transposition tables are saved periodically to disk, along with game statistics.
- **User Controls in GUI:** Buttons for Reset, Stop, and Save, plus CTRL+Q key shortcut for graceful quitting.

---

## Architecture and Code Structure

### Neural Network: ChessDQN

The network is defined in the `ChessDQN` class and is responsible for evaluating the board state. It takes a combined input vector representing the board and move information and outputs a single evaluation value. The architecture is:

- **Input:** 896 dimensions (768 from a 12×8×8 board tensor + 128 from move encoding)
- **Hidden Layers:** Two fully connected layers with 512 neurons each (with ReLU activations)
- **Output:** Single neuron for board evaluation

```python
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
```

### Board and Move Encoding

The program uses two helper functions to convert the board state and moves into numerical tensors:

- **`board_to_tensor(board)`**:  
  Encodes the chess board into a flattened array. It uses 12 channels—6 for each color—to represent the location of each type of piece (Pawn, Knight, Bishop, Rook, Queen, King) over an 8×8 grid.

- **`move_to_tensor(move)`**:  
  Encodes a move as a one-hot vector of length 128 (64 for the source square and 64 for the destination square).

```python
def board_to_tensor(board):
    # Create a 12x8x8 tensor (flattened to a 768-length vector)
    # Each channel corresponds to a specific piece type/color.

def move_to_tensor(move):
    # Create a 128-length one-hot vector
    #  - move.from_square -> index
    #  - move.to_square -> index+64
```

---

## Installation

### Prerequisites

- **Python 3.7+**
- **PyTorch:** For the neural network (supports CPU and GPU).
- **python-chess:** For board management and legal move generation.
- **Matplotlib:** For drawing the board and GUI elements.
- **NumPy:** For numerical computations.

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/chess-dqn-agent.git
   cd chess-dqn-agent
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install torch torchvision
   pip install python-chess matplotlib numpy
   ```

---

## Usage

Run the main program with:

```bash
python your_program.py
```

---

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes.
4. Push to your fork and open a pull request.

Please adhere to the code style and include comments where necessary.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Happy coding and enjoy exploring deep reinforcement learning in chess!*
