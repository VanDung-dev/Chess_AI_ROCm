import torch
import torch.nn as nn


class ChessNet(nn.Module):
    """
    Mạng nơ-ron cho trò chơi cờ vua, dự đoán policy (nước đi), value (giá trị bàn cờ) và giai đoạn trận đấu.

    Kiến trúc bao gồm CNN, khối tàn dư (Residual Blocks), và Transformer.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, kernel_size=3, padding=1)  # Đầu vào 16 kênh (đã thêm kênh giai đoạn)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(3)])
        self.embedding_dim = 64
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True),
            num_layers=3,
        )
        self.policy_head = nn.Linear(self.embedding_dim, 4672)  # Đầu ra policy
        self.value_head = nn.Linear(self.embedding_dim, 1)  # Đầu ra value
        self.stage_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Đầu ra giai đoạn: 3 lớp (khai cuộc, trung cuộc, tàn cuộc
        )

    def forward(self, x):
        """
        Lan truyền tiến qua mạng nơ-ron.

        Args:
            x (torch.Tensor): Tensor đầu vào (batch_size, 16, 8, 8).

        Returns:
            tuple: (policy, value, stage) - Xác suất nước đi, giá trị bàn cờ, và giai đoạn trận đấu.
        """
        x = torch.relu(self.conv(x))
        x = self.residual_blocks(x)
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        stage = self.stage_head(x)
        return policy, value, stage


class ResidualBlock(nn.Module):
    """
    Khối tàn dư (Residual Block) để cải thiện khả năng học của mạng sâu.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Lan truyền tiến qua khối tàn dư.

        Args:
            x (torch.Tensor): Tensor đầu vào.

        Returns:
            torch.Tensor: Tensor đầu ra sau khi thêm residual và kích hoạt.
        """
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)