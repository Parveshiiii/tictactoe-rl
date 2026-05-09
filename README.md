# Reinforcement Learning for Optimal Tic-Tac-Toe Strategy

This repository contains a high-performance Reinforcement Learning (RL) agent trained using a Dual-Update Self-Play methodology. The model has been trained through 800 million self-play iterations, achieving a state where it consistently reaches a draw against the optimal Minimax algorithm.

## Overview

While Tic-Tac-Toe is a mathematically solved game, this research explores achieving the optimal policy using an extremely minimal-parameter neural network. The objective is to demonstrate that a lightweight model can internalize a perfect strategy through 800 million iterations of self-play, resulting in a constant-time ($O(1)$) inference alternative to traditional search-based algorithms like Minimax.

>Note: This repo is structured using Gemini 

## Key Features

*   **Self-Play Training**: The agent learns solely through environmental interaction and self-competition.
*   **Massive Scale Simulation**: 800 million games played to ensure convergence and robustness.
*   **Dual-Update Mechanism**: A specialized training step that optimizes offensive and defensive strategies simultaneously.
*   **Provable Optimality**: Performance parity with the Minimax algorithm, resulting in forced draws.
*   **Ultra-Lightweight**: The serialized model weights are approximately **27 KB**, making it suitable for edge deployment.
*   **Inference Efficiency**: Unlike the Minimax algorithm, which requires an exhaustive tree search for every decision, this model determines the optimal move in a **single forward pass** ($O(1)$ complexity relative to the state space), providing significant computational advantages.

## Technical Architecture

The architecture consists of a multi-layer perceptron (MLP) enhanced with Layer Normalization and Residual Connections to stabilize training and improve gradient flow.

```python
class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        identity = x
        x = self.relu(self.ln2(self.fc2(x)))
        x = x + identity
        return self.fc3(x)
```

## Methodology

The training process utilizes a Policy Gradient approach. In each iteration, the model plays against an earlier version of itself. The "Dual-Update" strategy ensures that for every game state, the model is penalized or rewarded based on both its move quality and its ability to prevent the opponent from reaching a winning state.

## Installation and Usage

### Prerequisites

*   Python 3.8 or higher
*   PyTorch
*   Flask (for web interface)
*   Flask-CORS (for web interface)

### Installation

```bash
git clone https://github.com/Parveshiiii/tictactoe-rl.git
cd tictactoe-rl
pip install -r requirements.txt
```

### Execution

#### CLI Interface
To engage with the agent via the command line:
```bash
python play.py
```

#### Web Interface
To launch the browser-based interface:
1.  Initialize the backend server:
    ```bash
    python web/server.py
    ```
2.  Open `web/index.html` in a web browser.

#### Training
To execute the training pipeline:
```bash
python train.py
```

## License

This project is licensed under the MIT License.
