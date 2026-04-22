"""
Feature Extractor MLP cho dữ liệu NetFlow tabular
"""
import torch
import torch.nn as nn


class CNN1DFeatureExtractor(nn.Module):
    """
    1-D CNN for features extraction from NetFlow data.
    Architecture:
    Conv1d (k3, p0, s1) x 2 -> MaxPool1d (k2, s2) -> Conv1d (k3, p0, s1) x 2 -> AdaptiveMaxPool1d
    """

    def __init__(self, input_dim=41, output_dim=64):
        """
        Args:
            input_dim: Number of features (default 41)
            output_dim: Feature embedding dimension
        """
        super(CNN1DFeatureExtractor, self).__init__()
        
        # Architecture: 4 Conv1d layers (kernel 3, stride 1, padding 0)
        # Thứ tự chuẩn theo Fig 4: Conv1d -> ReLU -> BatchNorm1d
        self.body = nn.Sequential(
            # Lớp 1
            nn.Conv1d(1, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            
            # Lớp 2
            nn.Conv1d(32, 32, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Lớp 3
            nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Lớp 4
            nn.Conv1d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Global Pooling
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        out = self.body(x)
        out = out.view(out.size(0), -1) # Flatten (64 dimensions)
        return out


class LeNetTabular(nn.Module):
    """
    Mạng nhỏ dùng cho encode_model
    """

    def __init__(self, input_dim=43, hidden_dim=128, num_classes=10):
        super(LeNetTabular, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = self.fc(out)
        return out


def weights_init(m):
    """Khởi tạo trọng số ngẫu nhiên (tương tự GLFC)"""
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
    try:
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
