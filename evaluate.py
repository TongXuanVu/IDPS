"""
Đánh giá model HFIN: Accuracy, F1-score, Confusion Matrix, Forgetting
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, test_dataset, classes, device, batch_size=256):
    """
    Đánh giá model trên tập test
    
    Args:
        model: HFINNetwork
        test_dataset: NetFlowDataset
        classes: list hoặc [start, end] - phạm vi lớp cần test
        device: str
        batch_size: int
    
    Returns:
        dict với accuracy, f1_macro, f1_weighted, per_class_f1, y_true, y_pred
    """
    model.eval()
    model.to(device)

    test_dataset.getTestData(classes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0

    for _, features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(features)
            # DERNetwork tra ve (logits, aux_logits) -- chi can logits chinh
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item() * features.size(0)
        num_samples += features.size(0)

        preds = torch.max(outputs, dim=1)[1].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())


    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    avg_loss = total_loss / max(1, num_samples)

    # Tính Confusion Matrix để lấy FPR (False Positive Rate)
    # Trong NIDS, FPR = (Mẫu Benign bị đoán nhầm là Attack) / (Tổng mẫu Benign)
    # Giả định Class 0 là Benign (chuẩn CIC-IoT23)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)) if isinstance(classes, (list, range)) else None)
    fpr = 0.0
    if cm.shape[0] > 0:
        # TN = cm[0,0], FP = sum(cm[0, 1:])
        tn = cm[0, 0]
        fp = np.sum(cm[0, 1:])
        if (fp + tn) > 0:
            fpr = (fp / (fp + tn)) * 100

    results = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0) * 100,
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0) * 100,
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0) * 100,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        
        'fpr': fpr,
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0) * 100,
        'loss': avg_loss,
        'y_true': y_true,
        'y_pred': y_pred,
    }

    model.train()
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, dataset_name=None, title=None):
    """Vẽ và lưu confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    if not title:
        title = f'HFIN Confusion Matrix ({dataset_name})' if dataset_name else 'HFIN Confusion Matrix'
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curve(rounds, accuracies, save_path, dataset_name=None):
    """
    Vẽ accuracy qua các task
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Task')
    plt.ylabel('Accuracy (%)')
    title = f'HFIN Accuracy ({dataset_name})' if dataset_name else 'HFIN Accuracy'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_loss_curve(rounds, losses, save_path, dataset_name=None):
    """
    Vẽ loss qua các task
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(rounds, losses, 'r-o', linewidth=2, markersize=4)
    plt.xlabel('Task')
    plt.ylabel('Loss')
    title = f'HFIN Loss ({dataset_name})' if dataset_name else 'HFIN Loss'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics_curves(x_values, metrics_dict, save_path, xlabel='Task', dataset_name=None):
    """
    Vẽ nhiều chỉ số cùng lúc
    
    Args:
        x_values: list các giá trị trục X (Task ID hoặc Task.Progress)
        metrics_dict: dict {name: [values]}
        save_path: đường dẫn lưu file
        xlabel: nhãn trục X
        dataset_name: tên dataset để hiển thị trên title
    """
    plt.figure(figsize=(12, 7))
    
    styles = {
        'Accuracy': 'b-',
        'Precision': 'c-.',
        'Recall': 'm:',
        'Macro-F1': 'g--',
        'Weighted-F1': 'r:'
    }
    
    for name, values in metrics_dict.items():
        if not values: continue
        style = styles.get(name, '-')
        plt.plot(x_values, values, style, label=name, linewidth=2)
        plt.plot(x_values, values, style[0]+'o', markersize=4)

    plt.xlabel(xlabel)
    plt.ylabel('Score (%)')
    title = f'HFIN Performance ({dataset_name})' if dataset_name else 'HFIN Performance'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_forgetting(task_accuracies_per_class):
    """
    Tính forgetting metric: accuracy drop trên lớp cũ sau khi học lớp mới
    
    Args:
        task_accuracies_per_class: dict {global_round: {class_id: accuracy}}
    
    Returns:
        avg_forgetting: float
    """
    if len(task_accuracies_per_class) <= 1:
        return 0.0

    forgetting_values = []
    # global_round keys
    round_ids = sorted(task_accuracies_per_class.keys())

    # Duyệt qua từng lớp đã từng xuất hiện
    all_classes = set()
    for r in round_ids:
        all_classes.update(task_accuracies_per_class[r].keys())
    
    for class_id in all_classes:
        # Tìm round mà lớp này đạt accuracy cao nhất
        class_accuracies = []
        for r in round_ids:
            if class_id in task_accuracies_per_class[r]:
                class_accuracies.append(task_accuracies_per_class[r][class_id])
        
        if len(class_accuracies) > 1:
            max_acc = max(class_accuracies[:-1]) # Max của các lần trước
            final_acc = class_accuracies[-1]    # Lần cuối cùng
            forgetting_values.append(max_acc - final_acc)

    return np.mean(forgetting_values) if forgetting_values else 0.0


def print_evaluation_report(results, task_id, label_map=None, logger=None):
    """In báo cáo đánh giá chi tiết định dạng theo yêu cầu bài báo"""
    msg = []
    msg.append('\n' + '='*60)
    msg.append(f'  ĐÁNH GIÁ - Task {task_id}')
    msg.append('='*60)
    msg.append(f'  Accuracy:           {results["accuracy"]:.2f}%')
    
    msg.append(f'  Precision (micro):  {results["precision_micro"]:.2f}%')
    msg.append(f'  Precision (macro):  {results["precision_macro"]:.2f}%')
    msg.append(f'  Precision (weight): {results["precision_weighted"]:.2f}%')
    
    msg.append(f'  Recall (micro):     {results["recall_micro"]:.2f}%')
    msg.append(f'  Recall (macro):     {results["recall_macro"]:.2f}%')
    msg.append(f'  Recall (weight):    {results["recall_weighted"]:.2f}%')
    
    msg.append(f'  F1 (micro):         {results["f1_micro"]:.2f}%')
    msg.append(f'  F1 (macro):         {results["f1_macro"]:.2f}%')
    msg.append(f'  F1 (weight):        {results["f1_weighted"]:.2f}%')
    
    msg.append(f'  Loss:               {results["loss"]:.4f}')

    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
        msg.append(f'\n  Per-class F1:')
        for i, f1 in enumerate(results['per_class_f1']):
            name = inv_map.get(i, f'Class {i}')
            msg.append(f'    {name:20s}: {f1:.2f}%')

    msg.append('='*60 + '\n')
    
    report_text = '\n'.join(msg)
    if logger:
        logger.info(report_text)
    else:
        print(report_text)
