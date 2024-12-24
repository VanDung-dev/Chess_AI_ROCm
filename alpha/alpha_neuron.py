import torch
import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, 16)
        self.conv_input = nn.Conv2d(41, 20, kernel_size=1)
        self.conv = nn.Conv2d(20, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(4)])
        self.embedding_dim = 64
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8),
            num_layers=4,
            norm=None,
        )

        # Các đầu ra
        self.policy_head = nn.Linear(self.embedding_dim, 4672)
        self.value_head = nn.Linear(self.embedding_dim, 1)
        self.threat_head = nn.Conv2d(64, 1, kernel_size=1)
        self.phase_head = nn.Linear(self.embedding_dim, 3)
        self.strategy_head = nn.Linear(self.embedding_dim, 5)
        self.evaluation_head = nn.Linear(self.embedding_dim, 1)  # Đánh giá bàn cờ
        self.mobility_head = nn.Linear(self.embedding_dim, 1)  # Tính di động

    def forward(self, board, piece_types):
        piece_embedded = self.piece_embedding(piece_types).permute(0, 3, 1, 2)
        board_combined = torch.cat([board, piece_embedded], dim=1)
        board_combined = self.conv_input(board_combined)

        x = self.conv(board_combined)
        x = self.residual_blocks(x)

        attention_weights = torch.softmax(
            self.piece_embedding(piece_types).mean(dim=-1), dim=-1
        )
        x = x * attention_weights.unsqueeze(1)

        x_transformed = x.flatten(2).transpose(1, 2)
        x_transformed = x_transformed.permute(1, 0, 2)
        x_transformed = self.transformer(x_transformed)
        x_transformed = x_transformed.permute(1, 0, 2)

        x_mean = x_transformed.mean(dim=1)
        policy = self.policy_head(x_mean)
        value = self.value_head(x_mean)
        threats = self.threat_head(x)
        phase = self.phase_head(x_mean)
        strategy = self.strategy_head(x_mean)
        evaluation = self.evaluation_head(x_mean)
        mobility = self.mobility_head(x_mean)

        return policy, value, threats, phase, strategy, evaluation, mobility


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
