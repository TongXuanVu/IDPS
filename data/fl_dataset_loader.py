import os
import torch
import numpy as np

# Số clients theo FL partition mới
NUM_FL_CLIENTS = 10

# Phân bố class theo Task (đã tuần tự 0-33)
FL_TASK_CLASSES_SEQUENTIAL = {
    0: list(range(0, 6)),   # Task 1: 0,1,2,3,4,5
    1: list(range(6, 12)),  # Task 2: 6,7,8,9,10,11
    2: list(range(12, 18)), # Task 3: 12..17
    3: list(range(18, 24)), # Task 4: 18..23
    4: list(range(24, 29)), # Task 5 (5 classes): 24..28
    5: list(range(29, 34)), # Task 6 (5 classes): 29..33
}

# Biến tạm để tránh lỗi Import trên Kaggle nếu main.py hoặc file khác vẫn gọi
FL_TASK_CLASSES = {}
GLOBAL_LABEL_MAP = {}

# Số mẫu tối đa mỗi lớp trong tập Test (để tránh OOM/Crash trên Kaggle)
MAX_VAL_SAMPLES_PER_CLASS = 3000

def load_fl_global_test(data_dir):
    """
    Load tập test toàn cục trực tiếp và thực hiện downsampling để tiết kiệm RAM.
    """
    test_file = os.path.join(data_dir, "global_test_data.pt")
    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"[FL LOADER] Không tìm thấy file test toàn cục: {test_file}\n"
            f"Đảm bảo file 'global_test_data.pt' tồn tại trong thư mục data_split của FL."
        )

    print(f"[FL LOADER] Đang tải Global Test Set từ: {test_file}")
    data = torch.load(test_file, weights_only=False)
    if isinstance(data, dict):
        X, y = data['x'], data['y']
    else:
        X, y = data

    # ──────────────────────────────────────────────────────────────────
    # Downsampling tập Test để chống sập RAM (14M mẫu -> ~100k mẫu)
    # ──────────────────────────────────────────────────────────────────
    unique_classes = torch.unique(y)
    indices_to_keep = []
    
    for cls in unique_classes:
        cls_indices = (y == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) > MAX_VAL_SAMPLES_PER_CLASS:
            # Chọn ngẫu nhiên 3000 mẫu
            perm = torch.randperm(len(cls_indices))[:MAX_VAL_SAMPLES_PER_CLASS]
            indices_to_keep.append(cls_indices[perm])
        else:
            indices_to_keep.append(cls_indices)
    
    final_indices = torch.cat(indices_to_keep)
    X = X[final_indices]
    y = y[final_indices]
    
    print(f"[FL LOADER] Global Test (Optimized): {len(y):,} mẫu | Labels: {unique_classes.tolist()}")
    return X, y


def load_fl_client_task(data_dir, task_id, client_id):
    """
    Load file train .pt cho 1 client cụ thể (labels tuần tự).
    """
    task_num = task_id + 1
    filename = f"client_{client_id}_task_{task_num}.pt"
    client_file = os.path.join(data_dir, "federated_data", filename)

    if not os.path.exists(client_file):
        return None, None

    data = torch.load(client_file, weights_only=False)
    if isinstance(data, dict):
        X, y = data['x'], data['y']
    else:
        X, y = data

    return X, y


def update_clients_for_task(clients_dict, data_dir, task_id):
    """
    Cập nhật dữ liệu trong RAM của tất cả clients thành dữ liệu của Task hiện tại.
    """
    task_num = task_id + 1
    print(f"\n[FL LOADER] Đang tải dữ liệu Task {task_num} lên {len(clients_dict)} Clients...")

    for cid, client in clients_dict.items():
        X_train, y_train = load_fl_client_task(data_dir, task_id, cid)

        if X_train is None or len(X_train) == 0:
            client.train_data   = torch.zeros((0, 1), dtype=torch.float32)
            client.train_labels = torch.zeros(0, dtype=torch.long)
        else:
            client.train_data   = X_train
            client.train_labels = y_train
            unique_labels = torch.unique(y_train).tolist()
            print(f"   -> Client {cid:2d}: {len(X_train):6,} mẫu | Labels: {unique_labels}")

