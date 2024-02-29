import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITCHead(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
