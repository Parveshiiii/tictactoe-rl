import os
import torch
import torch.optim as optim
from model import TicTacToeModel

def check_winner(board):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for a, b, c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0.5  # Draw
    return None

def train():
    model = TicTacToeModel()
    checkpoint_path = "checkpoints/tictactoe.pth"
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    episodes = 100000
    
    print(f"{'Episode':<10} | {'Outcome':<8} | {'Avg Reward':<12}")
    print("-" * 40)

    recent_rewards = []

    for ep in range(episodes):
        board = [0] * 9
        done = False
        log_probs_p1, log_probs_p2 = [], []

        while not done:
            # Player 1
            state = torch.tensor(board, dtype=torch.float32)
            logits = model(state)
            mask = torch.tensor([board[i] != 0 for i in range(9)])
            logits[mask] = float("-inf")
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs_p1.append(dist.log_prob(action))
            board[action.item()] = 1

            result = check_winner(board)
            if result is not None:
                reward_p1 = 1.0 if result == 1 else 0.5
                done = True
            else:
                # Player 2
                inverted_board = [-v for v in board]
                state_opp = torch.tensor(inverted_board, dtype=torch.float32)
                logits_opp = model(state_opp)
                mask_opp = torch.tensor([board[i] != 0 for i in range(9)])
                logits_opp[mask_opp] = float("-inf")
                probs_opp = torch.softmax(logits_opp, dim=0)
                dist_opp = torch.distributions.Categorical(probs_opp)
                action_opp = dist_opp.sample()
                log_probs_p2.append(dist_opp.log_prob(action_opp))
                board[action_opp.item()] = -1

                result = check_winner(board)
                if result is not None:
                    reward_p1 = -1.0 if result == -1 else 0.5
                    done = True

        # Dual-Update Logic
        reward_p2 = -reward_p1 if reward_p1 in [1.0, -1.0] else 0.5
        policy_loss = [-lp * reward_p1 for lp in log_probs_p1] + [-lp * reward_p2 for lp in log_probs_p2]

        optimizer.zero_grad()
        sum(policy_loss).backward()
        optimizer.step()

        recent_rewards.append(reward_p1)
        if len(recent_rewards) > 100: recent_rewards.pop(0)

        if ep % 1000 == 0:
            avg = sum(recent_rewards) / len(recent_rewards)
            outcome = "Win" if reward_p1 == 1 else ("Draw" if reward_p1 == 0.5 else "Loss")
            print(f"{ep:<10} | {outcome:<8} | {avg:<12.2f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
