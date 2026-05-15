import os
import torch
import numpy as np
import pandas as pd

# Số lượng clients tổng cộng (3 Edges * 20 Clients) theo bài báo
NUM_FL_CLIENTS = 60
CLIENTS_PER_EDGE = 20

# Danh sách features chuẩn cho NF-UQ-NIDS-v2 (39 features từ PKL)
UQ_FEATURE_NAMES = [
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 
    'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 
    'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 
    'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 
    'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 
    'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
]

# Phân bố class theo Task cho các tập NetFlow
FL_TASK_CLASSES_SEQUENTIAL_UQ = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8],
    2: [9, 10, 11, 12],
    3: [13, 14, 15, 16],
    4: [17, 18, 19, 20],
}

FL_TASK_CLASSES_SEQUENTIAL_TON = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}

FL_TASK_CLASSES_SEQUENTIAL_CIC = {
    0: list(range(0, 6)),
    1: list(range(6, 12)),
    2: list(range(12, 18)),
    3: list(range(18, 24)),
    4: list(range(24, 29)),
    5: list(range(29, 34)),
}

def load_fl_global_test(data_dir, dataset_name):
    """Load tập test toàn cục tương ứng với dataset."""
    filename = f"global_test_{dataset_name}.pt"
    test_file_pt = os.path.join(data_dir, filename)
    
    if os.path.exists(test_file_pt):
        print(f"[FL LOADER] Loading Global Test Set: {test_file_pt}")
        data = torch.load(test_file_pt, weights_only=False)
        X, y = (data['x'], data['y']) if isinstance(data, dict) else data
        return X, y
    else:
        # Fallback to generic name if specific one not found
        generic_path = os.path.join(data_dir, "global_test_data.pt")
        if os.path.exists(generic_path):
            data = torch.load(generic_path, weights_only=False)
            return (data['x'], data['y']) if isinstance(data, dict) else data
            
    return None, None

def load_fl_client_task(data_dir, task_id, client_id, dataset_name):
    """
    Load file train cho 1 client cụ thể.
    """
    edge_id = client_id // CLIENTS_PER_EDGE
    intra_edge_id = client_id % CLIENTS_PER_EDGE
    
    task_folder = f"t={task_id + 1}"
    edge_filename_pq = f"client_{edge_id + 1}.parquet"
    
    # Map dataset_name to actual folder name
    ds_folder = "NF-ToN-IoT-v2" if dataset_name == 'nf_ton_iot' else "NF-UQ-NIDS-v2"
    full_path = os.path.join(data_dir, ds_folder, task_folder, edge_filename_pq)

    if os.path.exists(full_path):
        df = pd.read_parquet(full_path)
        label_col = 'class_id' if 'class_id' in df.columns else 'Label'
        
        # Chia dữ liệu của Edge thành 20 phần
        indices = np.arange(len(df))
        np.random.seed(42) 
        np.random.shuffle(indices)
        
        client_indices = np.array_split(indices, CLIENTS_PER_EDGE)[intra_edge_id]
        df_client = df.iloc[client_indices]
        
        # Label Shift
        y_raw = df_client[label_col].values
        y = torch.LongTensor(y_raw - 1)
        
        # Loại bỏ các cột phi số và nhãn để khớp 39 features
        drop_cols = [
            label_col, 'Label', 'label', 'Attack', 'class_id', 'Dataset',
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT'
        ]
        X_df = df_client.drop(columns=[c for c in drop_cols if c in df_client.columns])
        
        # Ensure only numeric
        for col in X_df.columns:
            if X_df[col].dtype == object:
                X_df = X_df.drop(columns=[col])
                
        X = torch.FloatTensor(X_df.values)
        return X, y

    return None, None


def update_clients_for_task(clients_dict, data_dir, task_id, dataset_name):
    """Cập nhật dữ liệu cho tất cả 60 clients."""
    print(f"\n[FL LOADER] Task {task_id + 1} ({dataset_name.upper()}) -> {len(clients_dict)} Clients...")
    for cid, client in clients_dict.items():
        X, y = load_fl_client_task(data_dir, task_id, cid, dataset_name)
        if X is not None:
            client.train_data, client.train_labels = X, y
        else:
            client.train_data = torch.zeros((0, 39))
            client.train_labels = torch.zeros(0, dtype=torch.long)


from typing import Dict, List
from collections import defaultdict

def assign_clients_to_edges(num_clients: int, num_edge_servers: int) -> Dict[int, List[int]]:
    edge_client_map = defaultdict(list)
    clients_per_edge = num_clients // num_edge_servers
    for eid in range(num_edge_servers):
        start = eid * clients_per_edge
        end = (eid + 1) * clients_per_edge if eid < num_edge_servers - 1 else num_clients
        edge_client_map[eid] = list(range(start, end))
    return edge_client_map
