"""
Mạng phân loại cho HFIN - Feature Extractor + Classification Head
Hỗ trợ mở rộng head khi có lớp tấn công mới (Class-Incremental)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class HFINNetwork(nn.Module):
    """
    Mạng kết hợp: Feature Extractor + Incremental Classifier
    Tương tự class `network` trong GLFC nhưng cho dữ liệu tabular
    """

    def __init__(self, num_classes, feature_extractor):
        """
        Args:
            num_classes: Số lớp ban đầu
            feature_extractor: Module trích xuất đặc trưng (MLPFeatureExtractor)
        """
        super(HFINNetwork, self).__init__()
        self.feature = feature_extractor

        # Lấy output dim từ feature extractor
        # Tìm lớp Linear cuối cùng trong body
        feature_dim = self._get_feature_dim()
        self.fc = nn.Linear(feature_dim, num_classes, bias=True)

    def _get_feature_dim(self):
        """Tìm output dimension của feature extractor"""
        # Nếu feature extractor có lớp fc (MLP cũ)
        if hasattr(self.feature, 'fc'):
            return self.feature.fc.out_features
            
        # Tìm lớp BatchNorm1d hoặc Conv1d cuối cùng (cho 1D-CNN)
        for module in reversed(list(self.feature.modules())):
            if isinstance(module, (nn.BatchNorm1d, nn.Conv1d)):
                return module.num_features if hasattr(module, 'num_features') else module.out_channels
        
        return 64 # Mặc định cho HFIN CNN

    def forward(self, x):
        """Forward pass: features → logits"""
        features = self.feature(x)
        logits = self.fc(features)
        return logits

    @property
    def out_features(self) -> int:
        """Tổng số classes hiện tại (API chung với DERNetwork)"""
        return self.fc.out_features


    def Incremental_learning(self, new_num_classes):
        """
        Mở rộng classification head cho lớp mới
        Giữ nguyên trọng số cũ, thêm neurons mới cho lớp mới
        """
        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data
        in_features = self.fc.in_features
        old_num_classes = self.fc.out_features

        # Tạo FC layer mới lớn hơn
        self.fc = nn.Linear(in_features, new_num_classes, bias=True)
        # Copy trọng số cũ
        self.fc.weight.data[:old_num_classes] = old_weight
        self.fc.bias.data[:old_num_classes] = old_bias

    def weight_align(self, old_num_classes, new_num_classes):
        """
        Weight Aligning (WA) - Eq. từ bài báo:
        Hiệu chỉnh trọng số của các lớp mới để tránh thiên kiến (bias)
        Norm(w_new) = Norm(w_old) * (mean_norm_old / mean_norm_new)
        """
        weights = self.fc.weight.data
        new_classes_count = new_num_classes - old_num_classes
        
        if old_num_classes == 0 or new_classes_count <= 0:
            return

        # Tính mean norm của các vector trọng số lớp cũ
        old_weights = weights[:old_num_classes]
        new_weights = weights[old_num_classes:new_num_classes]
        
        old_norms = torch.norm(old_weights, p=2, dim=1)
        new_norms = torch.norm(new_weights, p=2, dim=1)
        
        gamma = torch.mean(old_norms) / torch.mean(new_norms)
        
        # Cập nhật trọng số của các lớp mới
        self.fc.weight.data[old_num_classes:new_num_classes] *= gamma

    def feature_extractor(self, x):
        """Trích xuất đặc trưng (không qua FC head)"""
        return self.feature(x)

    def predict(self, features):
        """Phân loại từ features"""
        return self.fc(features)
