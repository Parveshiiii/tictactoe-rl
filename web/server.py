import os
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TicTacToeModel

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints/tictactoe.pth")

# Load model
model = TicTacToeModel()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

def check_winner(board):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for a, b, c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0.5
    return None

@app.route("/ai_move", methods=["POST"])
def ai_move():
    data = request.get_json()
    board = list(data["board"])
    
    if check_winner(board) is not None:
        return jsonify({"move": None, "board": board, "winner": check_winner(board)})

    with torch.no_grad():
        state = torch.tensor(board, dtype=torch.float32)
        logits = model(state)
        mask = torch.tensor([board[i] != 0 for i in range(9)])
        logits[mask] = float("-inf")
        action = torch.argmax(logits).item()

    board[action] = 1
    winner = check_winner(board)
    return jsonify({"move": action, "board": board, "winner": winner})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
