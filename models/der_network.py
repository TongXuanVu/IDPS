"""
DERNet - Dynamic Expansion Network cho Class-Incremental Learning
Port từ SPCIL/DER sang dữ liệu tabular (NetFlow) với CNN1DFeatureExtractor backbone

Architecture (giống SPCIL inc_net.py -> DERNet):
  convnets[0..T-1] : T CNN1D backbones, mỗi task thêm 1 backbone mới.
                     Backbone cũ bị đóng băng (frozen) khi học task mới.
  fc               : Main classifier over concat(all backbone features) -> total_classes
  aux_fc           : Auxiliary classifier chỉ trên features backbone mới nhất
                     -> new_task_classes + 1 (class 0 = background/old)

Loss mỗi incremental task:
  L = CE(fc_logits, targets) + lambda_aux * CE(aux_logits, aux_targets)
  aux_targets: relabel new classes 1..n, old-class samples -> 0
               (giống SPCIL: aux_targets - known_classes + 1 nếu > 0 else 0)

Weight Align: sau khi kết thúc mỗi incremental task (is_last_round=True)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import CNN1DFeatureExtractor


class DERNetwork(nn.Module):
    """
    Dynamic Expansion Network cho Class-Incremental Learning trên NetFlow data.
    Tương đương với DERNet trong SPCIL nhưng dùng CNN1DFeatureExtractor backbone.
    """

    def __init__(self, input_dim: int, feature_dim: int = 64):
        """
        Args:
            input_dim  : Số chiều đầu vào (số NetFlow features)
            feature_dim: Output dimension của mỗi CNN backbone
        """
        super().__init__()
        self.input_dim  = input_dim
        self.feature_dim = feature_dim      # out_dim mỗi backbone (như SPCIL out_dim)

        self.convnets   = nn.ModuleList()   # List of CNN1D backbones (1 per task)
        self.fc         = None              # Main classifier: total_features -> total_classes
        self.aux_fc     = None              # Auxiliary classifier: feature_dim -> new_task+1
        self.task_sizes = []                # Số classes mới mỗi task

    # ------------------------------------------------------------------
    @property
    def total_feature_dim(self) -> int:
        """Tổng số chiều features = feature_dim * số task đã học"""
        return self.feature_dim * len(self.convnets)

    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Concat features từ tất cả backbone"""
        feats = [convnet(x) for convnet in self.convnets]
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits     : (B, total_classes) — từ fc chính
            aux_logits : (B, new_task+1)   — từ aux_fc (backbone mới nhất)
        """
        if len(self.convnets) == 0:
            raise RuntimeError("DERNetwork: Chua co backbone. Goi update_fc truoc.")

        features     = self.extract_features(x)     # (B, feature_dim * T)
        logits       = self.fc(features)             # (B, total_classes)
        aux_features = self.convnets[-1](x)          # (B, feature_dim)
        aux_logits   = self.aux_fc(aux_features)     # (B, new_task+1)

        return logits, aux_logits

    @property
    def feature_extractor(self):
        """Alias cho extract_features đe ExemplarManager goi duoc"""
        # Tra ve mot function nhan x va tra ve feature
        return self.extract_features

    # ------------------------------------------------------------------
    def update_fc(self, total_classes: int):
        """
        Mở rộng mạng cho task mới (tương đương DERNet.update_fc):
          1. Thêm CNN backbone mới (copy weights từ backbone trước)
          2. Mở rộng fc chính (giữ trọng số cũ)
          3. Tạo mới aux_fc cho task này
        """
        # --- 1. Thêm backbone mới ---
        new_backbone = CNN1DFeatureExtractor(
            input_dim=self.input_dim,
            output_dim=self.feature_dim
        )
        if len(self.convnets) > 0:
            # Kế thừa trọng số từ backbone trước (giống SPCIL)
            new_backbone.load_state_dict(self.convnets[-1].state_dict())

        self.convnets.append(new_backbone)

        # --- 2. Mở rộng fc chính ---
        new_task_size = total_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        new_fc = nn.Linear(self.total_feature_dim, total_classes, bias=True)
        if self.fc is not None:
            # Giữ trọng số cũ (phần feature của các backbone cũ)
            old_out = self.fc.out_features
            old_in  = self.fc.in_features
            new_fc.weight.data[:old_out, :old_in] = self.fc.weight.data.clone()
            new_fc.bias.data[:old_out]             = self.fc.bias.data.clone()
        self.fc = new_fc

        # --- 3. Tạo aux_fc mới cho task này ---
        # Output: new_task_size + 1 (class 0 = background/old classes)
        self.aux_fc = nn.Linear(self.feature_dim, new_task_size + 1, bias=True)

    # ------------------------------------------------------------------
    def freeze_old_backbones(self):
        """Đóng băng tất cả backbone cũ (trừ backbone mới nhất)"""
        for convnet in self.convnets[:-1]:
            for param in convnet.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Mở khóa tất cả tham số (dùng khi load weights từ cloud)"""
        for param in self.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    def weight_align(self, increment: int):
        """
        Weight Aligning giữa lớp cũ và mới (giống SPCIL DERNet.weight_align):
          gamma = mean_norm(old_weights) / mean_norm(new_weights)
          new_weights *= gamma
        """
        if self.fc is None or increment <= 0:
            return
        weights = self.fc.weight.data
        if weights.shape[0] <= increment:
            return
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        if meannew > 1e-8:
            gamma = meanold / meannew
            self.fc.weight.data[-increment:, :] *= gamma

    # ------------------------------------------------------------------
    # API tương thích với HFINNetwork để main.py không cần thay đổi nhiều
    # ------------------------------------------------------------------
    def Incremental_learning(self, total_classes: int):
        """API tương thích với HFINNetwork.Incremental_learning"""
        self.update_fc(total_classes)

    @property
    def out_features(self) -> int:
        if self.fc is None:
            return 0
        return self.fc.out_features
