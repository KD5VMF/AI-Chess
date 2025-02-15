# Chess AI Self-Play and Human-vs-AI Training System

This project implements a chess-playing artificial intelligence (AI) that continuously learns and improves through self-play and human interaction. It features a deep Q-network (DQN) combined with a minimax search (with iterative deepening) to evaluate board positions, and offers multiple modes of operation including self-play (both fast and animated) and a graphical human-vs-AI interface complete with control buttons.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
  - [Neural Network](#neural-network)
  - [Board and Move Encoding](#board-and-move-encoding)
  - [Minimax with Iterative Deepening](#minimax-with-iterative-deepening)
  - [AI Agent and Training](#ai-agent-and-training)
- [Installation](#installation)
- [Usage](#usage)
  - [Modes of Operation](#modes-of-operation)
  - [Graphical Human-vs-AI Interface](#graphical-human-vs-ai-interface)
  - [Control Buttons](#control-buttons)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project is designed to create a chess AI that can learn from its own play as well as through interactions with human players. The AI uses a deep neural network to evaluate board positions and improve over time by training on game memories. Self-play is employed to generate training data continuously, while the human-vs-AI mode offers an interactive GUI for playing and learning.

## Features

- **Multiple Modes:**  
  - **Self-Play (Faster):** No board animation; optimized for rapid training.
  - **Self-Play (Slower):** Animated board to watch each move and monitor progress.
  - **Human-vs-AI (Graphical):** An interactive graphical interface with control buttons.
  
- **Deep Q-Network (DQN):**  
  A feed-forward neural network that evaluates board states.

- **Minimax with Iterative Deepening:**  
  A search algorithm that uses iterative deepening and a time cutoff to select moves.

- **Transposition Table:**  
  Caches board evaluations to speed up the search.

- **Statistics Tracking:**  
  Records wins, losses, draws, and global move counts for further analysis and training.

- **Control Buttons (Human-vs-AI Mode):**  
  Buttons for resetting the board, stopping the game, and saving progress (model, transposition table, and statistics).

- **Continuous Learning:**  
  The AI continually improves its play through self-play and human interaction. Training progress is saved periodically.

## System Architecture

### Neural Network

- **Architecture:**  
  The AI uses a simple feed-forward network with two hidden layers (512 units each) and ReLU activations. The network takes a flattened board state (768 values from 12 channels of an 8x8 board) concatenated with a move encoding (128 values) as input and outputs a single scalar evaluation.

### Board and Move Encoding

- **Board Encoding:**  
  The board is represented as a 12×8×8 tensor where each of the 12 channels represents one type of piece (6 for white, 6 for black). This tensor is flattened to form part of the input to the neural network.
  
- **Move Encoding:**  
  A move is encoded as a one-hot vector of length 128 (64 for the from-square and 64 for the to-square) and concatenated with the board tensor.

### Minimax with Iterative Deepening

- **Algorithm:**  
  The minimax algorithm is used with alpha-beta pruning and iterative deepening. A time cutoff is enforced to limit the search depth for AI moves.

- **Purpose:**  
  It provides move selection by exploring future board states and using the neural network to evaluate positions.

### AI Agent and Training

- **Agent Class:**  
  The `ChessAgent` class handles move selection, evaluation (with caching via the transposition table), and training. It uses an epsilon-greedy policy to balance exploration and exploitation.

- **Training Process:**  
  After each game, the agent trains on its stored game memory for a few epochs using mean squared error (MSE) loss between the predicted value and the game outcome.

## Installation

### Requirements

- **Python Version:**  
  Python 3.7 or higher is recommended.

- **Dependencies:**  
  - PyTorch
  - NumPy
  - python-chess
  - Matplotlib

### Installing Dependencies

Use `pip` to install the required packages:

```bash
pip install torch numpy python-chess matplotlib
