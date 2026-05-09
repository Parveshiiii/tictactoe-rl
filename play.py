import torch
from model import TicTacToeModel

def check_winner(board):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for a, b, c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0.5 if 0 not in board else None

def print_board(b):
    chars = {1: "X", -1: "O", 0: " "}
    for i in range(0, 9, 3):
        print(f" {chars[b[i]]} | {chars[b[i+1]]} | {chars[b[i+2]]} ")
        if i < 6: print("-----------")
    print()

def play():
    model = TicTacToeModel()
    try:
        model.load_state_dict(torch.load("checkpoints/tictactoe.pth"))
    except:
        print("Warning: No checkpoint found in 'checkpoints/tictactoe.pth'. Using untrained model.")
    
    model.eval()
    board = [0] * 9
    print("Welcome to Tic-Tac-Toe RL!")
    print("You are 'O' (-1), AI is 'X' (1).")

    while True:
        print_board(board)
        # Human Move
        move = -1
        while move not in range(9) or board[move] != 0:
            try: move = int(input("Enter move (0-8): "))
            except: continue
        board[move] = -1
        if check_winner(board) is not None: break

        # AI Move
        print("AI is thinking...")
        with torch.no_grad():
            state = torch.tensor(board, dtype=torch.float32)
            logits = model(state)
            mask = torch.tensor([board[i] != 0 for i in range(9)])
            logits[mask] = float("-inf")
            action = torch.argmax(logits).item()
            board[action] = 1
        if check_winner(board) is not None: break

    print_board(board)
    res = check_winner(board)
    print("AI Wins!" if res == 1 else ("You Won!" if res == -1 else "It's a Draw!"))

if __name__ == "__main__":
    play()
