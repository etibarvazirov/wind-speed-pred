import torch
import torch.nn as nn

# ============================================
# N-HiTS BLOCK
# ============================================
class NHiTSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ============================================
# FULL N-HiTS MODEL
# ============================================
class NHiTS(nn.Module):
    def __init__(self, seq_len, num_features=16, hidden_size=128, num_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        input_size = seq_len * num_features

        # Multiple N-HiTS blocks
        self.blocks = nn.ModuleList([
            NHiTSBlock(input_size, hidden_size, 1)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        b, s, f = x.shape
        x = x.reshape(b, s * f)  # flatten
        out = 0
        for block in self.blocks:
            out = out + block(x)
        return out.squeeze()
