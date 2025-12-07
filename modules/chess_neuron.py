import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    """
    Kiến trúc ResNet-SE nâng cao cho AI Cờ vua (Mục tiêu 2000+ Elo).

    Cấu trúc:
    - Lớp tích chập đầu vào (Input Conv)
    - Tháp các khối Residual với Squeeze-and-Excitation
    - Đầu ra Policy (Xác suất nước đi)
    - Đầu ra Value (Xác suất Thắng/Thua)
    """
    def __init__(self, input_channels=19, num_res_blocks=10, num_filters=128):
        super(ChessNet, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        
        # Initial Convolution
        # Lớp tích chập ban đầu
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Residual Tower
        # Tháp các khối Residual
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        # Đầu ra Policy
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672) # Kích thước nước đi tiêu chuẩn của AlphaZero
        
        # Value Head
        # Đầu ra Value
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Đầu vào: (Batch, Số kênh, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        # Đầu ra Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 8 * 8)
        p = self.policy_fc(p)
        # Lưu ý: Softmax thường được áp dụng trong hàm mất mát hoặc trong quá trình suy luận, ở đây trả về logits
        
        # Value Head
        # Đầu ra Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 32 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Giá trị đầu ra trong khoảng [-1, 1]
        
        return p, v
