"""
Tiện ích Federated Learning:
- FedAvg aggregation
- Model utilities
- Seed setup
- Training helpers
"""
import torch
import torch.nn as nn
import copy
import numpy as np
import random


def setup_seed(seed):
    """Thiết lập random seed cho reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_to_device(model, device):
    """Chuyển model sang device"""
    if device == 'cpu':
        model.to(torch.device('cpu'))
    else:
        model.to(torch.device(device))
    return model


def FedAvg(model_weights_list):
    """
    Federated Averaging - tổng hợp trọng số từ nhiều clients
    
    Args:
        model_weights_list: list of state_dict
    
    Returns:
        state_dict trung bình
    """
    w_avg = copy.deepcopy(model_weights_list[0])
    for key in w_avg.keys():
        for i in range(1, len(model_weights_list)):
            w_avg[key] += model_weights_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(model_weights_list))
    return w_avg


def FedWeightedAvg(model_weights_list, weights):
    """
    Weighted Federated Averaging - FedAvg có trọng số
    Dùng trong WTO (Weighted Transmission Optimization)
    
    Args:
        model_weights_list: list of state_dict
        weights: list of float - trọng số cho mỗi model
    
    Returns:
        state_dict trung bình có trọng số
    """
    total_weight = sum(weights)
    weights_normalized = [w / total_weight for w in weights]
    
    w_avg = copy.deepcopy(model_weights_list[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * weights_normalized[0]
        for i in range(1, len(model_weights_list)):
            w_avg[key] += model_weights_list[i][key] * weights_normalized[i]
    return w_avg


def model_global_eval(model, test_dataset, task_id, task_size, num_base_classes, device):
    """
    Đánh giá model global trên tất cả các lớp đã học
    
    Args:
        model: HFINNetwork
        test_dataset: NetFlowDataset
        task_id: int - task hiện tại
        task_size: int - số lớp mỗi task
        num_base_classes: int - số lớp base task
        device: str
    
    Returns:
        accuracy: float (phần trăm)
    """
    from torch.utils.data import DataLoader
    
    model = model_to_device(model, device)
    model.eval()
    
    # Tính tổng số lớp đã học
    if task_id == 0:
        total_classes = num_base_classes
    else:
        total_classes = num_base_classes + task_id * task_size
    
    test_dataset.getTestData([0, total_classes])
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=256)
    
    correct, total = 0, 0
    for _, features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(features)
            # DERNetwork tra ve (logits, aux_logits) -- chi can logits chinh
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts == labels).sum().item()
        total += len(labels)

    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    model.train()
    return accuracy


def get_task_classes(task_id, num_base_classes, task_size):
    """
    Lấy danh sách lớp cho task cụ thể
    
    Task 0: [0, 1, ..., num_base_classes-1]
    Task 1: [num_base_classes, num_base_classes+1, ..., num_base_classes+task_size-1]
    ...
    """
    if task_id == 0:
        return list(range(num_base_classes))
    else:
        start = num_base_classes + (task_id - 1) * task_size
        end = start + task_size
        return list(range(start, end))


def get_all_learned_classes(task_id, num_base_classes, task_size):
    """Lấy tất cả lớp đã học đến task hiện tại"""
    if task_id == 0:
        return list(range(num_base_classes))
    else:
        total = num_base_classes + task_id * task_size
        return list(range(total))
