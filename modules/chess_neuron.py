import torch
import torch.nn as nn


class ChessNet(nn.Module):
    """
    Mạng nơ-ron cho trò chơi cờ vua, dự đoán policy, value và giai đoạn trận đấu.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(20, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(3)])
        self.embedding_dim = 64
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, batch_first=True),
            num_layers=3,
        )
        self.policy_head = nn.Linear(self.embedding_dim, 4672)
        self.value_head = nn.Linear(self.embedding_dim, 1)
        self.stage_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.stage_adjust = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Lan truyền tiến qua mạng nơ-ron với tích hợp thông tin giai đoạn.

        Args:
            x (torch.Tensor): Tensor đầu vào (batch_size, 20, 8, 8).

        Returns:
            tuple: (policy, value, stage).
        """
        batch_size = x.size(0)
        stage_info = x[:, 15:16, 0, 0]
        stage_onehot = torch.zeros(batch_size, 3, device=x.device)
        for i in range(batch_size):
            if stage_info[i] < 0.33:
                stage_onehot[i, 0] = 1.0
            elif stage_info[i] < 0.66:
                stage_onehot[i, 1] = 1.0
            else:
                stage_onehot[i, 2] = 1.0

        x = torch.relu(self.conv(x))
        x = self.residual_blocks(x)
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)
        stage_features = self.stage_adjust(stage_onehot)
        x = x + stage_features
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        stage = self.stage_head(x)
        return policy, value, stage


class ResidualBlock(nn.Module):
    """
    Khối tàn dư để cải thiện khả năng học của mạng sâu.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)