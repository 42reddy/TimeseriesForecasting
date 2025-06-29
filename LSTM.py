import torch.nn as nn

class lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.ffn(last_hidden)

