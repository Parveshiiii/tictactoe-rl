import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import TicTacToeModel

# ==========================================
# 0. GPU SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on Device: {device.type.upper()} (Optimized for L4)")

# ==========================================
# 1. BATCHED Game Utilities (GPU Optimized)
# ==========================================
wins_tensor = torch.tensor(
    [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Vertical
        [0, 4, 8], [2, 4, 6]              # Diagonal
    ],
    device=device,
)

def check_winner_batched(boards):
    """Checks thousands of games at once without breaking graphs."""
    batch_size = boards.size(0)
    p1_win = torch.zeros(batch_size, dtype=torch.bool, device=device)
    p2_win = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(8):
        a, b, c = wins_tensor[i]
        p1_win |= (boards[:, a] == 1) & (boards[:, b] == 1) & (boards[:, c] == 1)
        p2_win |= (boards[:, a] == -1) & (boards[:, b] == -1) & (boards[:, c] == -1)

    draws = (boards == 0).sum(dim=1) == 0

    status = torch.zeros(batch_size, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    status[p1_win] = 1.0
    done[p1_win] = True

    status[p2_win & ~done] = -1.0
    done[p2_win & ~done] = True

    status[draws & ~done] = 0.5
    done[draws & ~done] = True

    return status, done

# ==========================================
# 2. Training Pipeline
# ==========================================
def train():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    model = TicTacToeModel().to(device)
    SAVE_PATH = "checkpoints/tictactoe_l4_extreme_scratch.pth"
    
    # Training Hyperparameters for NVIDIA L4
    batch_size = 32768  # Massive parallel simulation
    training_steps = 100000 
    lr = 0.002

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"🌱 Initializing fresh model. Training for ~81.9 Million Games.")
    print(f"\n{'Step':<5} | {'Games Played':<12} | {'P1 Win%':<7} | {'Draw%':<7} | {'P2 Win%':<7} | {'Temp'}")
    print("-" * 75)

    for step in range(training_steps):
        boards = torch.zeros((batch_size, 9), device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        log_probs_p1_list = []
        log_probs_p2_list = []
        entropy_p1_list = []
        entropy_p2_list = []
        final_status = torch.zeros(batch_size, device=device)

        progress = step / training_steps
        temperature = max(1.0, 3.0 - 2.0 * progress)
        entropy_coef = 0.5 * (0.01**progress)

        for turn in range(9):
            if not active.any(): break

            player = 1 if turn % 2 == 0 else -1
            state = boards[active] if player == 1 else -boards[active]

            logits = model(state)
            invalid_moves = boards[active] != 0
            logits = logits.masked_fill(invalid_moves, float("-inf"))

            probs = torch.softmax(logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            boards[active, action] = player

            if player == 1:
                log_probs_p1_list.append((dist.log_prob(action), active.clone()))
                entropy_p1_list.append((dist.entropy(), active.clone()))
            else:
                log_probs_p2_list.append((dist.log_prob(action), active.clone()))
                entropy_p2_list.append((dist.entropy(), active.clone()))

            status, done = check_winner_batched(boards)
            newly_done = done & active
            final_status[newly_done] = status[newly_done]
            active = active & ~done

        # Dual Reward & Safe Loss Calculation
        reward_p1 = final_status.clone()
        reward_p2 = torch.zeros_like(final_status)
        reward_p2[final_status == 1.0] = -1.0
        reward_p2[final_status == -1.0] = 1.0
        reward_p2[final_status == 0.5] = 0.5

        adv_p1 = reward_p1 - reward_p1.mean()
        adv_p2 = reward_p2 - reward_p2.mean()

        policy_loss = 0
        entropy_loss = 0

        for log_prob, mask in log_probs_p1_list:
            policy_loss -= (log_prob * adv_p1[mask]).sum()
        for ent, mask in entropy_p1_list:
            entropy_loss -= (entropy_coef * ent).sum()

        for log_prob, mask in log_probs_p2_list:
            policy_loss -= (log_prob * adv_p2[mask]).sum()
        for ent, mask in entropy_p2_list:
            entropy_loss -= (entropy_coef * ent).sum()

        total_loss = (policy_loss + entropy_loss) / batch_size

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == training_steps - 1:
            p1_win_pct = (final_status == 1.0).float().mean().item() * 100
            draw_pct = (final_status == 0.5).float().mean().item() * 100
            p2_win_pct = (final_status == -1.0).float().mean().item() * 100
            total_games = (step + 1) * batch_size
            print(f"{step:<5} | {total_games:<12,} | {p1_win_pct:>6.1f}% | {draw_pct:>6.1f}% | {p2_win_pct:>6.1f}% | {temperature:<5.2f}")

        if step % 500 == 0 and step > 0:
            torch.save(model.state_dict(), SAVE_PATH)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n🔥 Extreme Batched L4 Training Complete! Saved as {SAVE_PATH}")

if __name__ == "__main__":
    train()
