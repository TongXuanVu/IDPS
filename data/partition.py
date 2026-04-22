"""
Phân chia dữ liệu theo thông số bài báo HFIN (Mục VI.B)

Thông số chính:
  - Train/Test: 60% / 40%
  - Non-IID: Phân phối Dirichlet (α = 0.5)
  - 3 Edge Server, mỗi Edge 20 Client (tổng 60 Client)
  - Task schedule:
      NF-ToN-IoT-v2  (10 class): base=4, incremental=[2,2,2] hoặc [5]  → 4,2,2,2 | 4,5,1
      NF-UQ-NIDS-v2  (21 class): base=1, incremental=[2]*10 | [4]*5 | [10]*2
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


# =====================================================================
# Cấu hình Task Schedule theo bài báo (Mục VI.B, Table I)
# =====================================================================
TASK_CONFIGS = {
    # --- NF-ToN-IoT-v2: 10 class (1 Benign + 9 attack) ---
    'nf_ton_iot': {
        'total_classes': 10,
        'schedules': {
            'task2': {'base': 2, 'step': 2, 'num_tasks': 4},   # 2 + 2*4 = 10
            'task5': {'base': 5, 'step': 5, 'num_tasks': 1},   # 5 + 5 = 10
        },
        'default_schedule': 'task2'
    },
    # --- NF-UQ-NIDS-v2: 20 class (1 Benign + 19 attack) ---
    'nf_uq_nids': {
        'total_classes': 20,
        'schedules': {
            'task2':  {'base': 2, 'step': 2,  'num_tasks': 9},   # 2 + 2*9 = 20
            'task4':  {'base': 4, 'step': 4,  'num_tasks': 4},   # 4 + 4*4 = 20
            'task10': {'base': 10, 'step': 10, 'num_tasks': 1},  # 10 + 10 = 20
        },
        'default_schedule': 'task2'
    },
    # --- CIC-IoT23: 34 class | 6 tasks sequential (FL partition mới) ---
    # Task 1-4: 6 nhãn mỗi task | Task 5-6: 5 nhãn mỗi task
    # Labels: [0-5] → [6-11] → [12-17] → [18-23] → [24-28] → [29-33]
    'cic_iot23': {
        'total_classes': 34,
        'schedules': {
            'task6': {'base': 6, 'step': 6, 'num_tasks': 5},   # 6 + 6*3 + 5*2 = 34 (approx)
        },
        'default_schedule': 'task6'
    }
}


def get_task_schedule(dataset_name: str, schedule_key: str = None) -> List[List[int]]:
    """
    Tạo danh sách các class cho từng task theo bài báo.

    Args:
        dataset_name: tên dataset
        schedule_key: 'task2', 'task4', 'task5', 'task10' (None = dùng default)

    Returns:
        tasks: list of lists, mỗi phần tử là danh sách class trong 1 task
               tasks[0] = base task, tasks[1..] = incremental tasks
    """
    cfg = TASK_CONFIGS[dataset_name]
    key = schedule_key or cfg['default_schedule']
    sched = cfg['schedules'][key]

    base    = sched['base']
    step    = sched['step']
    n_tasks = sched['num_tasks']
    total   = cfg['total_classes']

    tasks = [list(range(base))]                    # base task
    for i in range(n_tasks):
        start = base + i * step
        end   = min(start + step, total)
        if start >= total:
            break
        tasks.append(list(range(start, end)))

    return tasks


def partition_data_non_iid(
    y_train: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Phân chia dữ liệu NON-IID dùng phân phối Dirichlet (Mục VI.B).
    
    Args:
        y_train     : nhãn tập train
        num_clients : tổng số client (60 theo bài báo)
        alpha       : float hoặc Dict[int, float] cho từng lớp.
                      Bài báo: 0.3 cho Benign (lớp 0), 0.8 cho Attack.
        seed        : random seed
    """
    np.random.seed(seed)
    all_classes = np.unique(y_train)
    
    # Tạo index theo lớp
    class_indices = {c: np.where(y_train == c)[0] for c in all_classes}
    for c in all_classes:
        np.random.shuffle(class_indices[c])

    client_data_indices: Dict[int, List[int]] = defaultdict(list)

    for c in all_classes:
        idx = class_indices[c]
        n   = len(idx)
        if n == 0: continue

        # Xác định alpha cho lớp này theo Mục VI.B
        # Benign (lớp 0): 0.3 | Attack (lớp > 0): 0.8
        if isinstance(alpha, dict):
            current_alpha = alpha.get(int(c), 0.5)
        else:
            # Nếu truyền float đơn lẻ, áp dụng logic mặc định của bài báo
            # Giả định lớp 0 là Benign
            current_alpha = 0.3 if c == 0 else 0.8

        # Dirichlet proportions cho num_clients
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * current_alpha)
        splits = (proportions * n).astype(int)
        splits[-1] = n - splits[:-1].sum()   # Bù sai số làm tròn

        cursor = 0
        for client_id, size in enumerate(splits):
            if size > 0:
                client_data_indices[client_id].extend(idx[cursor:cursor + size].tolist())
                cursor += size

    # Chuyển sang np.array và tính client_classes
    client_data_out: Dict[int, np.ndarray] = {}
    client_classes: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        arr = np.array(client_data_indices[cid])
        client_data_out[cid] = arr
        if len(arr) > 0:
            client_classes[cid] = np.unique(y_train[arr]).tolist()
        else:
            client_classes[cid] = []

    return client_data_out, client_classes


def partition_data_by_task(
    y_train: np.ndarray,
    task_classes: List[int],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Phân chia non-IID cho dữ liệu của 1 task cụ thể.

    Chỉ xét các mẫu thuộc task_classes, sau đó chạy Dirichlet.

    Args:
        y_train      : toàn bộ nhãn train
        task_classes : danh sách lớp trong task hiện tại
        num_clients  : số client tham gia task này
        alpha        : Dirichlet
        seed         : random seed

    Returns:
        client_data_indices : dict {client_id: indices}
        client_classes      : dict {client_id: classes}
    """
    task_mask    = np.isin(y_train, task_classes)
    task_indices = np.where(task_mask)[0]
    y_task       = y_train[task_indices]

    local_idx, local_cls = partition_data_non_iid(y_task, num_clients, alpha, seed)

    # Map local index → global index trong y_train
    global_idx  = {cid: task_indices[local_idx[cid]] for cid in local_idx}
    return global_idx, local_cls


def assign_clients_to_edges(
    num_clients: int,
    num_edge_servers: int
) -> Dict[int, List[int]]:
    """
    Phân chia client vào edge server.
    Theo bài báo: n=3 edges, m=20 clients/edge → chia đều theo khối.
    """
    edge_client_map: Dict[int, List[int]] = defaultdict(list)
    clients_per_edge = num_clients // num_edge_servers
    
    for eid in range(num_edge_servers):
        start = eid * clients_per_edge
        end = (eid + 1) * clients_per_edge if eid < num_edge_servers - 1 else num_clients
        edge_client_map[eid] = list(range(start, end))
        
    return dict(edge_client_map)


def print_partition_stats(
    client_data_indices: Dict[int, np.ndarray],
    client_classes: Dict[int, List[int]],
    y_train: np.ndarray,
    edge_client_map: Dict[int, List[int]] = None
):
    """In thống kê về phân chia dữ liệu"""
    print('\n=== Thống kê phân chia dữ liệu ===')
    total = sum(len(v) for v in client_data_indices.values())
    print(f'Tổng mẫu phân chia: {total}')

    if edge_client_map:
        for eid, cids in sorted(edge_client_map.items()):
            edge_total = sum(len(client_data_indices[c]) for c in cids)
            edge_classes = set()
            for c in cids:
                edge_classes.update(client_classes[c])
            print(f'\nEdge {eid}: {len(cids)} clients | {edge_total} samples | classes: {sorted(edge_classes)}')
            for cid in cids[:3]:   # In 3 client đầu làm ví dụ
                n = len(client_data_indices[cid])
                cls = client_classes[cid]
                print(f'  Client {cid:2d}: {n:6d} samples | classes: {cls}')
            if len(cids) > 3:
                print(f'  ... ({len(cids) - 3} clients more)')
    else:
        for cid in range(min(5, len(client_data_indices))):
            n = len(client_data_indices[cid])
            cls = client_classes[cid]
            print(f'  Client {cid}: {n} samples | classes: {cls}')
