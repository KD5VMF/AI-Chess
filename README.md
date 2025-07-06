# ♟️ AI Chess Trainer with GPU Acceleration

Welcome to **AI-Chess**, an advanced self-learning chess engine powered by PyTorch, GPU acceleration, and modern reinforcement learning techniques. This project has been upgraded to include a deeper neural network, improved action selection, and curriculum learning.

---

## 🚀 Features

- ✅ **GPU Acceleration** — Uses PyTorch with CUDA to maximize NVIDIA GPU performance.
- ♟️ **Self-Play Training** — DQN + MCTS-style hybrid self-play, alternating white/black control.
- 🧠 **Deeper Neural Network** — Uses LayerNorm, GELU activation, Dropout, and additional layers.
- 📈 **Curriculum Learning** — Starts with easier AI opponents and gradually increases difficulty.
- 🧮 **Smarter Action Selection** — Combines deep Q-learning with Monte Carlo Tree Search.
- 👤 **Human vs AI Mode** — Play against the AI using a GUI chess board.
- 📉 **Debug & Logging** — Optional extended logging for performance tracking.

---

## 🗂️ Folder Structure

```
AI-Chess/
│
├── AI-Chess-Advanced-Final2.py   # Main upgraded Python file
├── requirements_ai_chess.txt     # Dependency list
├── models/                       # Saved models for white/black
├── tables/                       # Experience replay or MCTS tables
└── assets/                       # Icons, sounds, or board images
```

---

## 🖥️ Requirements

- Python 3.8+
- NVIDIA GPU with CUDA 11.8 or newer
- Linux/Ubuntu 20.04/22.04/24.04 LTS recommended

Install dependencies with:

```bash
pip install -r requirements_ai_chess.txt
```

---

## 🧪 Usage

Start the program with:

```bash
python3 AI-Chess-Advanced-Final2.py
```

Then choose from the menu:

- [1] Self-play training (Faster)
- [2] Self-play training (with GUI)
- [3] Human vs AI (playable GUI)
- [4] Toggle debug logging

---

## 🧠 AI Model

The model now includes:

- `Linear(input_size, 1024)`
- `GELU` activation
- `LayerNorm`
- `Dropout(p=0.3)`
- 4-layer deep network
- GPU tensor acceleration with `model.to(device)`

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 🙋‍♂️ Contributions

PRs welcome! Fork the project, test, improve the training pipeline or add new features.

---

**Game on. Train smarter. Play stronger.**
