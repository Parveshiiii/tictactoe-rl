import torch
import torch.nn as nn

class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()
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
