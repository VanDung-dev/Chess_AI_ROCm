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
        # Thêm layer để xử lý giai đoạn
        self.stage_adjust = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Lan truyền tiến qua mạng nơ-ron với tích hợp thông tin giai đoạn.

        Args:
            x (torch.Tensor): Tensor đầu vào (batch_size, 16, 8, 8).

        Returns:
            tuple: (policy, value, stage) - Xác suất nước đi, giá trị bàn cờ, và giai đoạn trận đấu.
        """
        # Tách thông tin giai đoạn từ kênh cuối cùng
        batch_size = x.size(0)
        stage_info = x[:, -1:, 0, 0]  # Lấy giá trị giai đoạn từ góc bàn cờ
        stage_onehot = torch.zeros(batch_size, 3, device=x.device)

        # Chuyển giá trị giai đoạn thành one-hot encoding
        for i in range(batch_size):
            if stage_info[i] < 0.33:  # Khai cuộc
                stage_onehot[i, 0] = 1.0
            elif stage_info[i] < 0.66:  # Trung cuộc
                stage_onehot[i, 1] = 1.0
            else:  # Tàn cuộc
                stage_onehot[i, 2] = 1.0

        # Xử lý CNN
        x = torch.relu(self.conv(x))
        x = self.residual_blocks(x)

        # Xử lý Transformer
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling

        # Kết hợp thông tin giai đoạn
        stage_features = self.stage_adjust(stage_onehot)
        x = x + stage_features  # Thêm đặc trưng giai đoạn vào embedding

        # Đầu ra policy (nước đi)
        policy = self.policy_head(x)

        # Đầu ra value (giá trị bàn cờ)
        value = torch.tanh(self.value_head(x))  # Giới hạn trong [-1, 1]

        # Đầu ra stage (dự đoán lại giai đoạn để kiểm tra)
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