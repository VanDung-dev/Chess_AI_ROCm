import torch
import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(13, 64, kernel_size=3, padding=1)  # 13: số loại quân cờ
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(3)]
        )
        self.embedding_dim = 64
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8),
            num_layers=3
        )
        self.policy_head = nn.Linear(self.embedding_dim, 4672)  # 4672 nước đi hợp lệ
        self.value_head = nn.Linear(self.embedding_dim, 1)  # Giá trị trạng thái

    def forward(self, board):
        x = self.conv(board)
        x = self.residual_blocks(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ChessEmbedding(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.piece_embedding = nn.Embedding(num_embeddings=13, embedding_dim=embedding_dim)  # 13 loại quân cờ
        self.position_embedding = nn.Embedding(num_embeddings=64, embedding_dim=embedding_dim)  # 64 ô bàn cờ

    def forward(self, board):
        """Mã hóa bàn cờ thành tensor (8x8) chứa chỉ số của các quân cờ (0-12)."""
        piece_features = self.piece_embedding(board)
        position_features = self.position_embedding(torch.arange(64).view(8, 8).to(board.device))
        return piece_features + position_features

class ChessTransformer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=12, num_layers=3):
        super().__init__()
        self.embedding = ChessEmbedding(embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embedding_dim, 1)  # Dự đoán giá trị

class ChessNetWithThreats(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = ChessNet()
        self.threat_head = nn.Linear(64, 64)  # Dự đoán quân cờ bị đe dọa

    def forward(self, board):
        policy, value = self.base_model(board)
        threat_scores = self.threat_head(board.flatten(1))
        return policy, value, threat_scores

class ResidualChessNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=5):
        super().__init__()
        self.input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return self.output_layer(x)

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

