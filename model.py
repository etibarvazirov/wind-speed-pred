import torch
import torch.nn as nn

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

class NHiTS(nn.Module):
    def __init__(self, seq_len, num_features, hidden_size=128, num_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        
        input_size = seq_len * num_features
        self.blocks = nn.ModuleList([
            NHiTSBlock(input_size, hidden_size, 1)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        b, s, f = x.shape
        x = x.reshape(b, s * f)
        
        y = None
        for block in self.blocks:
            if y is None:
                y = block(x)
            else:
                y = y + block(x)

        return y  # DO NOT SQUEEZE
