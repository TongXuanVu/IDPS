"""
HFIN - Main Training Pipeline
Hierarchical Federated Class-Incremental Learning for Network Intrusion Detection

Luồng huấn luyện:
1. Load & tiền xử lý dataset
2. Khởi tạo: Cloud Server, Edge Servers, Clients
3. Với mỗi global round:
   a. Phân phối global model → Cloud → Edge
   b. Edge Servers thu thập dữ liệu từ Clients và huấn luyện local (WTO)
   c. Edge Servers tổng hợp weights nội bộ rồi gửi lên Cloud
   d. Cloud Server tổng hợp global (FedAvg)
   e. Đánh giá global model
"""
import os
import sys
import copy
import random
import numpy as np
import torch
import logging
from datetime import datetime

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import args_parser
from data.preprocessing import load_and_preprocess
from data.partition import get_task_schedule, partition_data_by_task, partition_data_non_iid, assign_clients_to_edges
from data.dataset import NetFlowDataset
from models.feature_extractor import CNN1DFeatureExtractor, LeNetTabular, weights_init
from models.network import HFINNetwork
from models.der_network import DERNetwork
from federated.client import HFINClient
from federated.edge_server import EdgeServer
from federated.cloud_server import CloudServer
from federated.fed_utils import (
    setup_seed, model_to_device, FedAvg,
    model_global_eval, get_task_classes, get_all_learned_classes
)
from evaluate import (
    evaluate_model, plot_confusion_matrix, 
    print_evaluation_report, plot_metrics_curves
)


def setup_logging(log_dir, args):
    """Thiết lập logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Thiết lập logic logging chứa cả ngày và giờ
    run_folder = datetime.now().strftime("%d-%m-%y_%H-%M")
    
    # Tạo thư mục theo cấu trúc logs/dd-mm-yy_HH-MM/
    run_log_dir = os.path.join(args.log_dir, run_folder)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Cập nhật args.log_dir để các hàm sau (như evaluate plot) tự động lưu chung vào folder này
    args.log_dir = run_log_dir
    
    log_file = os.path.join(run_log_dir, "training.log")
    
    # Đảm bảo stdout xử lý được Unicode trên Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('HFIN')


def main():
    # === Parse arguments ===
    args = args_parser()
    
    # === Setup ===
    setup_seed(args.seed)
    logger = setup_logging(args.log_dir, args)
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info('='*60)
    logger.info('HFIN - Hierarchical Federated Class-Incremental Learning')
    logger.info('   Network Intrusion Detection for IIoT')
    logger.info('='*60)
    logger.info(f'Device: {args.device}')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Method: {args.method.upper()}')
    logger.info(f'Clients: {args.num_clients}, Edge Servers: {args.num_edge_servers}')
    logger.info(f'Base classes: {args.num_base_classes}, Task size: {args.task_size}')
    logger.info(f'Total classes: {args.total_classes}')
    
    # === Load & Tiền xử lý dữ liệu tự động ===
    logger.info('\n[1/4] Loading and preprocessing dataset...')
    # Sử dụng logic gốc
    X_train, X_test, y_train, y_test, scaler, label_map = load_and_preprocess(
        data_path=args.data_path,
        dataset_name=args.dataset,
        test_size=args.test_size,
        random_state=args.seed,
        max_samples=args.max_samples
    )
    
    num_features = X_train.shape[1]
    args.total_classes = len(label_map)
    
    logger.info(f'Features: {num_features}, Global Test: {len(X_test)}')
    logger.info(f'Total classes: {args.total_classes} | Clients: {args.num_clients} | Edges: {args.num_edge_servers}')
    
    # Test dataset (dùng chung cho đánh giá)
    test_dataset = NetFlowDataset(X_test, y_test)
    
    # === Phân chia dữ liệu cho clients ===
    logger.info(f'\n[2/4] Mapping {args.num_clients} FL Clients to {args.num_edge_servers} Edge Server(s)...')
    
    # === Phân chia dữ liệu cho clients ===
    logger.info(f'\n[2/4] Mapping {args.num_clients} FL Clients to {args.num_edge_servers} Edge Servers...')
    edge_client_map = assign_clients_to_edges(args.num_clients, args.num_edge_servers)
    logger.info(f'Edge-Client mapping: {edge_client_map}')
    
    # === Khoi tao models & components ===
    logger.info('\n[3/4] Initializing models...')
    feature_extractor = CNN1DFeatureExtractor(
        input_dim=num_features,
        output_dim=args.feature_dim
    )

    # Global model: DERNetwork cho DER, HFINNetwork cho iCaRL/WA
    if args.method in ('der', 'der++'):
        model_g = DERNetwork(input_dim=num_features, feature_dim=args.feature_dim)
        model_g.update_fc(args.num_base_classes)   # them backbone base
        logger.info(f'[DER] DERNetwork initialized: 1 backbone, {args.num_base_classes} classes')
    else:
        model_g = HFINNetwork(args.num_base_classes, copy.deepcopy(feature_extractor))
    model_g = model_to_device(model_g, args.device)

    # Tao clients (Data Providers)
    clients_dict = {}
    for c_id in range(args.num_clients):
        client = HFINClient(
            client_id=c_id,
            train_data=np.array([]),
            train_labels=np.array([]),
            device=args.device
        )
        clients_dict[c_id] = client

    # Tao edge servers (Local Trainers)
    edge_servers = []
    for e_id in range(args.num_edge_servers):
        edge = EdgeServer(
            edge_id=e_id,
            num_classes=args.num_base_classes,
            feature_extractor=copy.deepcopy(feature_extractor),
            device=args.device,
            memory_size=args.memory_size,
            task_size=args.task_size,
            method=args.method,
            der_alpha=args.der_alpha,
            der_beta=args.der_beta,
            max_samples_per_class=args.max_samples_per_class,
            downsample_ratio=args.downsample_ratio,
            input_dim=num_features,
            feature_dim=args.feature_dim,
        )
        edge.set_clients(edge_client_map[e_id])
        edge_servers.append(edge)

    # === Cloud server: dung model_g de init ===
    # DER: CloudServer chi luu model_g lam reference; FedAvg chay binh thuong
    encode_model = LeNetTabular(
        input_dim=num_features,
        hidden_dim=128,
        num_classes=args.total_classes
    )
    encode_model.apply(weights_init)
    cloud_server = CloudServer(
        num_classes=args.num_base_classes,
        feature_extractor=copy.deepcopy(feature_extractor),
        device=args.device,
        learning_rate=args.learning_rate,
        encode_model=copy.deepcopy(encode_model)
    )
    
    # === Training Loop (Task-based) ===
    logger.info('\n[4/4] Starting training...')
    accuracy_history = []
    precision_history = []
    recall_history = []
    f1_macro_history = []
    f1_weighted_history = []
    loss_history = []
    eval_round_history = []
    task_progress_history = []
    
    task_accuracies_per_class = {}  # Lưu accuracy từng lớp theo round để tính forgetting
    classes_learned = args.num_base_classes
    current_f1_scores = {i: 0.9 for i in range(args.total_classes)}
    global_round = 0  # Bộ đếm round tổng (để log)

    # Phân chia Task tự động theo bài báo
    all_task_classes = get_task_schedule(args.dataset, args.task_schedule)
    num_tasks = len(all_task_classes)
    
    # Đảo ngược label_map để in report đẹp hơn
    inv_label_map = {v: k for k, v in label_map.items()}
    # Nếu label_map có string keys, ta sẽ dùng trực tiếp cho report

    for task_id, task_classes in enumerate(all_task_classes):
        logger.info(f'\nPartitioning data for Task {task_id} (Classes: {task_classes})...')
        
        # Phân chia dữ liệu train cho các clients theo task
        client_data_indices, _ = partition_data_by_task(
            y_train=y_train,
            task_classes=task_classes,
            num_clients=args.num_clients,
            alpha=None, # Tự động dùng 0.3 cho benign, 0.8 cho attack
            seed=args.seed + task_id
        )
        
        # Đẩy dữ liệu vào Clients
        for cid, client in clients_dict.items():
            idx = client_data_indices.get(cid, [])
            if len(idx) > 0:
                client.train_data = torch.tensor(X_train[idx], dtype=torch.float32)
                client.train_labels = torch.tensor(y_train[idx], dtype=torch.long)
            else:
                client.train_data = torch.zeros((0, num_features), dtype=torch.float32)
                client.train_labels = torch.zeros(0, dtype=torch.long)
        
        # Rút gọn dữ liệu cho debug mode
        if args.debug and args.max_samples > 0:
            for cid, client in clients_dict.items():
                if len(client.train_data) > args.max_samples:
                    client.train_data = client.train_data[:args.max_samples]
                    client.train_labels = client.train_labels[:args.max_samples]
        
        # Số rounds cho task này (Sec VI.B)
        num_rounds = args.epochs_base if task_id == 0 else args.epochs_incremental

        # Cập nhật số lớp đã học
        classes_learned = len(task_classes) if task_id == 0 else classes_learned + len(task_classes)

        # Mo rong model: DERNetwork them backbone moi, HFINNetwork mo rong fc head
        if classes_learned > model_g.out_features:
            if isinstance(model_g, DERNetwork):
                # DER: them 1 backbone CNN moi cho task nay (dynamic expansion)
                model_g.update_fc(classes_learned)
                logger.info(
                    f'[DER] Added backbone {len(model_g.convnets)}, '
                    f'total features={model_g.total_feature_dim}, '
                    f'classes={classes_learned} for Task {task_id}'
                )
            else:
                # iCaRL / WA: chi mo rong fc head
                model_g.Incremental_learning(classes_learned)
                logger.info(f'Expanded global model to {classes_learned} classes for Task {task_id}')

        # Đồng bộ kiến trúc sang Cloud & Edge
        model_g = model_to_device(model_g, args.device)
        cloud_server.model = copy.deepcopy(model_g)
        for edge in edge_servers:
            edge.model = copy.deepcopy(model_g)
            edge.learned_classes = list(range(classes_learned))

        # Learning Rate theo Sec VI.B
        current_lr = args.learning_rate if task_id == 0 else args.lr_incremental
        logger.info(f'\n{"="*60}')
        logger.info(f'Task {task_id}/{num_tasks - 1} | Classes: {task_classes} | '
                    f'{num_rounds} rounds | LR: {current_lr}')
        logger.info(f'{"="*60}')

        # === Inner loop: rounds cho task này ===
        for round_in_task in range(num_rounds):
            global_round += 1

            logger.info(f'\n--- Task {task_id}, Round {round_in_task + 1}/{num_rounds} '
                        f'(Global {global_round}) ---')

            # Phân phối Global Model về các Edges
            global_weights = cloud_server.get_weights()
            for edge in edge_servers:
                edge.set_weights(global_weights)

            # Xác định sớm các cờ vòng lặp cần thiết cho Edge training
            is_first_round = (round_in_task == 0 and task_id == 0)
            is_eval_round = (round_in_task + 1) % args.eval_interval == 0
            is_last_round_of_task = (round_in_task == num_rounds - 1)

            # === Edge Training (WTO + Data Collection + FCIL) ===
            edge_weights = []
            edge_samples = []

            for edge in edge_servers:
                weights, num_samples = edge.train_local(
                    clients_dict=clients_dict,
                    global_round=global_round,
                    task_id=task_id,
                    task_classes=task_classes,
                    current_f1_scores=current_f1_scores,
                    epochs=args.epochs_local,
                    lr=current_lr,
                    batch_size=args.batch_size,
                    is_last_round=is_last_round_of_task
                )
                if weights:
                    edge_weights.append(weights)
                    edge_samples.append(num_samples)

            # === Cloud-level Global Aggregation ===
            if edge_weights:
                global_weights_new = cloud_server.aggregate_from_edges(edge_weights, edge_samples)
                model_g.load_state_dict(global_weights_new)

            # Cập nhật cloud server state
            cloud_server.model.load_state_dict(model_g.state_dict())

            if is_first_round or is_eval_round or is_last_round_of_task:
                results_eval = evaluate_model(model_g, test_dataset, range(classes_learned), args.device)
                acc = results_eval['accuracy']
                precision = results_eval['precision']
                recall = results_eval['recall']
                f1_macro = results_eval['f1_macro']
                f1_weighted = results_eval['f1_weighted']
                loss = results_eval['loss']
                
                accuracy_history.append(acc)
                precision_history.append(precision)
                recall_history.append(recall)
                f1_macro_history.append(f1_macro)
                f1_weighted_history.append(f1_weighted)
                loss_history.append(loss)
                eval_round_history.append(global_round)
                
                # Tính toán Task progress: task_id + (round_hien_tai / tong_so_round_cua_task)
                # Ví dụ: Task 0 kết thúc ở round 2/2 -> task_progress = 0 + 2/2 = 1.0
                task_progress = task_id + (round_in_task + 1) / num_rounds
                task_progress_history.append(task_progress)
                
                logger.info(
                    f'  [Eval] Task {task_id}, R {round_in_task + 1}: '
                    f'Acc: {acc:.2f}% | '
                    f'Prec: {precision:.2f}% | '
                    f'Recall: {recall:.2f}% | '
                    f'Ma-F1: {f1_macro:.2f}% | '
                    f'We-F1: {f1_weighted:.2f}% | '
                    f'Loss: {loss:.4f}'
                )

                # Lưu accuracy phục vụ tính Forgetting
                task_accuracies_per_class[global_round] = {
                    i: f1 for i, f1 in enumerate(results_eval['per_class_f1'])
                }

                # Cập nhật F1 scores cho WTO ở round tiếp theo (không ghi đè lớp cũ)
                for i, f1 in enumerate(results_eval['per_class_f1']):
                    current_f1_scores[i] = float(f1)

        # === SAU KHI KẾT THÚC CÁC VÒNG CỦA TASK: WA (Weight Aligning) ===
        # Theo Section IV.C: WA là cơ chế căn chỉnh trọng số cho iCaRL, chạy với method 'wa'
        if task_id > 0 and args.method == 'wa':
            # Diagnostic: Log weight norms before WA
            with torch.no_grad():
                weights = model_g.fc.weight.data
                old_norm = torch.norm(weights[:classes_learned-len(task_classes)], p=2, dim=1).mean().item()
                new_norm = torch.norm(weights[classes_learned-len(task_classes):classes_learned], p=2, dim=1).mean().item()
                logger.info(f'  [WA-Pre] Avg Norm: Old Classes = {old_norm:.4f}, New Classes = {new_norm:.4f}')

            logger.info(f'  [WA] Applying Weight Aligning for Task {task_id}...')
            model_g.weight_align(classes_learned - len(task_classes), classes_learned)
            
            # Diagnostic: Log weight norms after WA
            with torch.no_grad():
                weights = model_g.fc.weight.data
                old_norm_post = torch.norm(weights[:classes_learned-len(task_classes)], p=2, dim=1).mean().item()
                new_norm_post = torch.norm(weights[classes_learned-len(task_classes):classes_learned], p=2, dim=1).mean().item()
                logger.info(f'  [WA-Post] Avg Norm: Old Classes = {old_norm_post:.4f}, New Classes = {new_norm_post:.4f}')

            # Đánh giá lại sau WA để xem hiệu quả
            acc_post = model_global_eval(
                model_g, test_dataset, task_id, args.task_size,
                args.num_base_classes, args.device
            )
            # Cập nhật lại cloud server với các trọng số đã được align
            cloud_server.model.load_state_dict(model_g.state_dict())
            
            # Đánh giá lại sau WA để có F1 scores chuẩn cho task report cuối task
            # Chỉ in report này nếu chưa phải task cuối (để tránh lặp vì đã có FINAL report sau loop)
            final_task_results = evaluate_model(model_g, test_dataset, range(classes_learned), args.device)
            if task_id < num_tasks - 1:
                print_evaluation_report(final_task_results, task_id, label_map, logger)

    # === Kết thúc: Đánh giá cuối cùng & Forgetting Metric ===
    from evaluate import compute_forgetting
    avg_forgetting = compute_forgetting(task_accuracies_per_class)
    
    # Tính Macro-F1 cuối cùng
    final_classes_all = list(range(min(classes_learned, args.total_classes)))
    final_results = evaluate_model(model_g, test_dataset, final_classes_all, args.device)
    
    logger.info(f'\n{"="*60}')
    logger.info(f'  KẾT QUẢ CUỐI CÙNG')
    logger.info(f'  Final Accuracy:      {final_results["accuracy"]:.2f}%')
    logger.info(f'  Final Macro-F1:      {final_results["f1_macro"]:.2f}%')
    logger.info(f'  Final Weighted-F1:   {final_results["f1_weighted"]:.2f}%')
    logger.info(f'  Avg Forgetting:      {avg_forgetting:.2f}%')
    logger.info(f'{"="*60}')
    logger.info('\n' + '='*60)
    logger.info('TRAINING COMPLETED!')
    logger.info('='*60)
    
    # Đánh giá chi tiết cuối cùng (Final Summary)
    final_classes = list(range(min(classes_learned, args.total_classes)))
    results = evaluate_model(model_g, test_dataset, final_classes, args.device)
    print_evaluation_report(results, "FINAL", label_map, logger)
    
    # Lưu metrics mỗi round ra CSV
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(args.log_dir, f'metrics_round_by_round_{args.method}.csv')
    df = pd.DataFrame({
        'Global_Round': eval_round_history,
        'Task_Progress': task_progress_history,
        'Accuracy': accuracy_history,
        'Precision': precision_history,
        'Recall': recall_history,
        'Macro-F1': f1_macro_history,
        'Weighted-F1': f1_weighted_history,
        'Loss': loss_history
    })
    df.to_csv(csv_path, index=False)
    logger.info(f'Đã lưu tiến độ huấn luyện vào CSV: {csv_path}')
    
    # Hàm vẽ biểu đồ riêng biệt cho từng loại metric để dễ so sánh
    def save_single_plot(x_vals, y_vals, metric_name, color, marker):
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
        plt.xlabel('Task Progression')
        plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
        plt.title(f'[{args.method.upper()}] {metric_name} over Tasks ({args.dataset})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = metric_name.lower().replace("-", "_")
        plt.savefig(os.path.join(args.log_dir, f'{args.method}_{safe_name}.png'), dpi=150)
        plt.close()

    # Vẽ riêng từng metric
    save_single_plot(task_progress_history, accuracy_history, 'Accuracy', 'b', 'o')
    save_single_plot(task_progress_history, precision_history, 'Precision', 'c', 's')
    save_single_plot(task_progress_history, recall_history, 'Recall', 'm', 'd')
    save_single_plot(task_progress_history, f1_macro_history, 'Macro-F1', 'g', '^')
    save_single_plot(task_progress_history, f1_weighted_history, 'Weighted-F1', 'r', 'v')
    save_single_plot(task_progress_history, loss_history, 'Loss', 'k', 'X')
    
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map.get(i, f'Class {i}') for i in final_classes]
    plot_confusion_matrix(
        results['y_true'], results['y_pred'], class_names,
        os.path.join(args.log_dir, 'confusion_matrix.png'),
        dataset_name=args.dataset,
        title=f'HFIN Confusion Matrix ({args.dataset}) - Final Task'
    )
    
    # Lưu model
    save_path = os.path.join(args.save_dir, 'hfin_final_model.pth')
    torch.save({
        'model_state_dict': model_g.state_dict(),
        'args': vars(args),
        'accuracy_history': accuracy_history,
        'final_results': {
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted']
        }
    }, save_path)
    logger.info(f'Model saved: {save_path}')


if __name__ == '__main__':
    main()
