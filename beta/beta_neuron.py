import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(3)])
        self.embedding_dim = 64
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, nhead=8, batch_first=True
            ),
            num_layers=3,
        )
        self.policy_head = nn.Linear(self.embedding_dim, 4672)
        self.value_head = nn.Linear(self.embedding_dim, 1)

    def forward(self, board):
        x = self.conv(board)
        x = self.residual_blocks(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
