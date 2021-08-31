import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, input_size, head_n, dropout, batch_first=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(input_size, head_n, dropout=dropout, batch_first=batch_first)
            for _ in range(4)])

    def forward(self, x: torch.tensor):
        logits = x.unsqueeze(1)
        for layer in self.attentions:
            logits = self.batch_norm(logits.squeeze(1)).unsqueeze(1)
            logits, _ = layer(logits, logits, logits)

        return logits


class Transformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, head_n=10):
        super().__init__()
        self.placeholder = head_n - input_size % head_n
        input_size = input_size + self.placeholder
        self.attentions = SelfAttention(input_size, head_n, dropout=0)
        self.linear_relu_stack2 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        place_holder_tensor = torch.zeros(x.shape[0], self.placeholder).to(x.device)
        x = torch.cat([x, place_holder_tensor], axis=1)
        logits = self.attentions(x).squeeze(1)
        logits = self.linear_relu_stack2(logits)
        return logits
