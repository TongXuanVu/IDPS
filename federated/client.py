import torch
import numpy as np
from data.dataset import NetFlowDataset

class HFINClient:
    """
    Federated Client cho HFIN - Đóng vai trò Data Provider (IIoT Device).
    Tài nguyên hạn chế, chỉ thu thập và cung cấp dữ liệu cho Edge Server.
    """

    def __init__(self, client_id, train_data, train_labels, device='cpu'):
        """
        Args:
            client_id   : ID của client
            train_data  : Tensor hoặc ndarray (N, features)
            train_labels: Tensor hoặc ndarray (N,)
            device      : str
        """
        self.client_id    = client_id
        self.device       = device
        self.train_data   = train_data
        self.train_labels = train_labels
        self.dataset      = NetFlowDataset(train_data, train_labels)

    def _labels_numpy(self):
        """Trả về train_labels dưới dạng np.ndarray (tương thích cả Tensor và ndarray)."""
        if isinstance(self.train_labels, torch.Tensor):
            return self.train_labels.numpy()
        return np.array(self.train_labels)

    def get_class_counts(self):
        """Trả về phân bố lớp dữ liệu hiện tại (phục vụ WTO tại Server)."""
        labels = self._labels_numpy()
        if len(labels) == 0:
            return {}
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))

    def get_data_for_edge(self, task_classes=None):
        """
        Cung cấp dữ liệu thô (flows) cho Edge Server.
        Nếu task_classes được chỉ định, chỉ trả về dữ liệu của các lớp đó.

        Returns:
            (X, y) cặp torch.FloatTensor / torch.LongTensor
        """
        # Chuẩn hóa sang Tensor
        if isinstance(self.train_data, torch.Tensor):
            X = self.train_data.float()
        else:
            X = torch.FloatTensor(self.train_data)

        if isinstance(self.train_labels, torch.Tensor):
            y = self.train_labels.long()
        else:
            y = torch.LongTensor(self.train_labels)

        if len(X) == 0:
            return X, y

        if task_classes is None:
            return X, y

        # Lọc dữ liệu theo task
        mask = torch.zeros(len(y), dtype=torch.bool)
        for cls in task_classes:
            mask |= (y == cls)

        return X[mask], y[mask]

    def __len__(self):
        return len(self.train_data) if hasattr(self.train_data, '__len__') else 0

