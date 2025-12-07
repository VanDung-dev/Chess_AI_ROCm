import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(nn.Module):
    """
    Mạng nơ-ron cho trò chơi cờ vua, dự đoán policy, value và giai đoạn trận đấu.
    
    Attributes:
        conv (nn.Conv2d): Lớp tích chập đầu vào để trích xuất đặc trưng cục bộ.
        residual_blocks (nn.Sequential): Chuỗi các khối residual để học biểu diễn sâu.
        transformer (nn.TransformerEncoder): Bộ mã hóa transformer để học biểu diễn toàn cục.
        policy_head (nn.Linear): Đầu ra dự đoán policy (nước đi).
        value_head (nn.Linear): Đầu ra dự đoán value (đánh giá vị trí).
        stage_head (nn.Sequential): Đầu ra dự đoán stage (giai đoạn trận đấu).
        stage_adjust (nn.Sequential): Điều chỉnh đặc trưng theo thông tin stage.
    """
    def __init__(self, input_channels=30, embedding_dim=64, num_residual_blocks=3, 
                 num_transformer_layers=3, num_heads=8, dropout_rate=0.1):
        """
        Khởi tạo mạng ChessNet.
        
        Args:
            input_channels (int): Số lượng channels đầu vào (mặc định: 30).
            embedding_dim (int): Kích thước embedding cho transformer (mặc định: 64).
            num_residual_blocks (int): Số lượng khối residual (mặc định: 3).
            num_transformer_layers (int): Số lượng lớp transformer (mặc định: 3).
            num_heads (int): Số lượng heads trong transformer (mặc định: 8).
            dropout_rate (float): Tỷ lệ dropout để tránh overfitting (mặc định: 0.1).
        """
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        
        # Lớp tích chập đầu vào
        self.conv = nn.Conv2d(input_channels, embedding_dim, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm2d(embedding_dim)
        
        # Các khối residual
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(embedding_dim, embedding_dim, dropout_rate) 
            for _ in range(num_residual_blocks)
        ])
        
        # Transformer encoder để học biểu diễn toàn cục
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )
        
        # Đầu ra
        self.policy_head = nn.Linear(embedding_dim, 4672)
        self.value_head = nn.Linear(embedding_dim, 1)
        self.stage_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3)
        )
        self.stage_adjust = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Lan truyền tiến qua mạng nơ-ron với tích hợp thông tin giai đoạn.

        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, input_channels, 8, 8).

        Returns:
            tuple: (policy, value, stage) - các dự đoán cho policy, value và stage.
            
        Raises:
            ValueError: Nếu kích thước đầu vào không hợp lệ.
        """
        # Kiểm tra kích thước đầu vào
        if x.dim() != 4 or x.shape[1] != self.input_channels or x.shape[2:] != (8, 8):
            raise ValueError(f"Expected input shape (batch_size, {self.input_channels}, 8, 8), got {x.shape}")
        
        batch_size = x.size(0)
        
        # Trích xuất thông tin stage và chuyển đổi thành one-hot vector
        stage_info = x[:, 15:16, 0, 0].squeeze(1)  # (batch_size,)
        
        # Vector hóa việc tạo one-hot encoding để tăng hiệu suất
        stage_indices = (stage_info * 3).long().clamp(0, 2)  # Chuyển đổi về indices 0, 1, 2
        stage_onehot = F.one_hot(stage_indices, num_classes=3).float()  # (batch_size, 3)
        stage_onehot = stage_onehot.to(x.device)
        
        # Truyền qua mạng chính
        x = F.relu(self.conv_bn(self.conv(x)))
        x = self.residual_blocks(x)
        
        # Reshape cho transformer: (batch_size, 64, embedding_dim)
        x = x.view(batch_size, 64, self.embedding_dim)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Tích hợp thông tin stage
        stage_features = self.stage_adjust(stage_onehot)  # (batch_size, 64)
        x = x + stage_features  # Cộng đặc trưng
        x = self.dropout(x)
        
        # Đầu ra
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        stage = self.stage_head(x)
        return policy, value, stage


class ResidualBlock(nn.Module):
    """
    Khối tàn dư để cải thiện khả năng học của mạng sâu.
    
    Attributes:
        conv1 (nn.Conv2d): Lớp tích chập đầu tiên trong khối.
        bn1 (nn.BatchNorm2d): Chuẩn hóa batch sau conv1.
        conv2 (nn.Conv2d): Lớp tích chập thứ hai trong khối.
        bn2 (nn.BatchNorm2d): Chuẩn hóa batch sau conv2.
        dropout (nn.Dropout): Lớp dropout để tránh overfitting.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        """
        Khởi tạo khối residual.
        
        Args:
            in_channels (int): Số lượng channels đầu vào.
            out_channels (int): Số lượng channels đầu ra.
            dropout_rate (float): Tỷ lệ dropout để tránh overfitting.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        """
        Lan truyền tiến qua khối residual.
        
        Args:
            x (torch.Tensor): Tensor đầu vào.
            
        Returns:
            torch.Tensor: Tensor đầu ra sau khi đi qua khối residual.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)