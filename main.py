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
from data.fl_dataset_loader import (
    load_fl_global_test, update_clients_for_task, 
    FL_TASK_CLASSES_SEQUENTIAL_CIC, FL_TASK_CLASSES_SEQUENTIAL_NF,
    assign_clients_to_edges
)
from data.dataset import NetFlowDataset
from models.feature_extractor import CNN1DFeatureExtractor, LeNetTabular, weights_init
from models.network import HFINNetwork
from models.der_network import DERNetwork
from federated.client import HFINClient
from federated.edge_server import EdgeServer
from federated.cloud_server import CloudServer
from federated.fed_utils import (
    setup_seed, model_to_device, FedAvg, FedWeightedAvg
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



# =============================================================================
# TEST MODE — Phan code rieng biet chi de chay evaluation
# Cach su dung:
#   python main.py --mode test --method icarl
#   python main.py --mode test --method wa --test_checkpoint_dir ./checkpoints/run_xyz
# =============================================================================
def _run_test_mode(args, logger):
    """
    Load tung checkpoint da luu trong qua trinh train va chay evaluation.
    Ket qua duoc luu vao:
      - args.log_dir/test_results_<method>.csv   (tat ca checkpoint)
      - args.log_dir/test_results_<method>.log   (log chi tiet)
    """
    import csv as _csv
    import glob

    # Xac dinh thu muc checkpoint can doc
    ckpt_root = args.test_checkpoint_dir or args.checkpoint_dir
    if not os.path.isdir(ckpt_root):
        logger.error(f'[TEST] Khong tim thay thu muc checkpoint: {ckpt_root}')
        return

    # Tim tat ca file checkpoint cua method nay, sap xep theo global_round
    pattern = os.path.join(ckpt_root, f'ckpt_{args.method}_*.pth')
    ckpt_files = sorted(glob.glob(pattern))
    if not ckpt_files:
        logger.error(f'[TEST] Khong co checkpoint nao khop voi pattern: {pattern}')
        return

    logger.info(f'[TEST] Tim thay {len(ckpt_files)} checkpoint(s) trong: {ckpt_root}')

    # Load global test set
    logger.info('[TEST] Loading global test data...')
    X_test, y_test = load_fl_global_test(args.data_path)
    from data.dataset import NetFlowDataset
    test_dataset = NetFlowDataset(X_test.numpy(), y_test.numpy())
    
    # Cấu hình schedule theo dataset
    from data.fl_dataset_loader import FL_TASK_CLASSES_SEQUENTIAL_UQ, FL_TASK_CLASSES_SEQUENTIAL_TON, FL_TASK_CLASSES_SEQUENTIAL_CIC
    if args.dataset == 'nf_uq_nids':
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_UQ
    elif args.dataset == 'nf_ton_iot':
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_TON
    else:
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_CIC

    label_map = {i: f'Class_{i}' for i in range(args.total_classes)}
    num_features = X_test.shape[1]

    # Khoi tao model (kien truc)
    from models.feature_extractor import CNN1DFeatureExtractor, LeNetTabular, weights_init
    from models.network import HFINNetwork
    from models.der_network import DERNetwork

    # Khởi tạo model sẽ được thực hiện trong vòng lặp dựa vào task_id của từng checkpoint

    # History for plotting
    accuracy_history = []
    prec_mic_history = []
    prec_mac_history = []
    prec_wei_history = []
    rec_mic_history = []
    rec_mac_history = []
    rec_wei_history = []
    f1_mic_history = []
    f1_mac_history = []
    f1_wei_history = []
    loss_history = []
    global_round_history = []

    # CSV dau ra
    out_csv = os.path.join(args.log_dir, f'test_results_{args.method}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = _csv.writer(fcsv)
        writer.writerow([
            'checkpoint', 'task_id', 'round_in_task', 'global_round', 'classes_learned',
            'acc',
            'prec_mic', 'prec_mac', 'prec_wei',
            'rec_mic',  'rec_mac',  'rec_wei',
            'f1_mic',   'f1_mac',   'f1_wei',
            'fpr', 'loss',
        ])

        for ckpt_path in ckpt_files:
            ckpt_name = os.path.basename(ckpt_path)
            logger.info(f'[TEST] Evaluating: {ckpt_name}')

            state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            task_id        = state.get('task_id', -1)
            round_in_task  = state.get('round_in_task', -1)
            global_round   = state.get('global_round', -1)
            classes_learned = state.get('classes_learned', args.total_classes)

            # Re-instantiate model to match checkpoint architecture
            ckpt_out_classes = state['model_state_dict']['fc.weight'].shape[0]
            ckpt_in_features = state['model_state_dict']['fc.weight'].shape[1]
            num_backbones_in_ckpt = ckpt_in_features // args.feature_dim

            if args.method in ('der', 'der++'):
                model_g = DERNetwork(input_dim=num_features, feature_dim=args.feature_dim)
                curr_cls = 0
                for t in range(num_backbones_in_ckpt):
                    if t < len(curr_schedule):
                        curr_cls += len(curr_schedule[t])
                    else:
                        curr_cls = ckpt_out_classes
                    model_g.update_fc(curr_cls)
            else:
                feature_extractor = CNN1DFeatureExtractor(input_dim=num_features, output_dim=args.feature_dim)
                model_g = HFINNetwork(ckpt_out_classes, copy.deepcopy(feature_extractor))

            model_g.to(args.device)

            # --- Lọc bỏ keys bị lệch (Chống crash như lỗi aux_fc) ---
            model_state = model_g.state_dict()
            ckpt_state = state['model_state_dict']
            filtered_state = {}
            for k, v in ckpt_state.items():
                if k in model_state and v.shape != model_state[k].shape:
                    logger.warning(f"Bỏ qua key {k} do lệch kích thước: {v.shape} vs {model_state[k].shape}")
                    continue
                filtered_state[k] = v
                
            model_g.load_state_dict(filtered_state, strict=False)
            model_g.eval()

            # Danh gia
            results = evaluate_model(
                model_g, test_dataset,
                range(ckpt_out_classes), args.device
            )
            acc      = results['accuracy']
            f1_mic   = results['f1_micro']
            f1_mac   = results['f1_macro']
            f1_wei   = results['f1_weighted']
            loss     = results['loss']
            prec_mic = results.get('precision_micro', 0)
            prec_mac = results.get('precision_macro', 0)
            prec_wei = results.get('precision_weighted', 0)
            rec_mic  = results.get('recall_micro', 0)
            rec_mac  = results.get('recall_macro', 0)
            rec_wei  = results.get('recall_weighted', 0)

            logger.info(
                f"[TEST] {ckpt_name} | Acc: {acc:.2f}% | F1-Mac: {f1_mac:.2f}% | F1-Wei: {f1_wei:.2f}% | FPR: {results.get('fpr', 0):.2f}%"
            )

            writer.writerow([
                ckpt_name, task_id, round_in_task, global_round, classes_learned,
                round(acc, 4),
                round(prec_mic, 4), round(prec_mac, 4), round(prec_wei, 4),
                round(rec_mic, 4),  round(rec_mac, 4),  round(rec_wei, 4),
                round(f1_mic, 4),   round(f1_mac, 4),   round(f1_wei, 4),
                round(results.get('fpr', 0), 4),
                round(loss, 6),
            ])
            fcsv.flush()

            # Save to history for plotting
            accuracy_history.append(acc)
            prec_mic_history.append(prec_mic)
            prec_mac_history.append(prec_mac)
            prec_wei_history.append(prec_wei)
            rec_mic_history.append(rec_mic)
            rec_mac_history.append(rec_mac)
            rec_wei_history.append(rec_wei)
            f1_mic_history.append(f1_mic)
            f1_mac_history.append(f1_mac)
            f1_wei_history.append(f1_wei)
            loss_history.append(loss)
            global_round_history.append(global_round if global_round != -1 else len(accuracy_history))

    logger.info(f'[TEST] Done! Ket qua luu tai: {out_csv}')

    # --- Ve bieu do cho tung metric (tuong tu SPCIL) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    def save_test_plot(x_vals, y_vals, metric_name, color, marker):
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
        plt.xlabel('Global Round / Checkpoint Index')
        plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
        plt.title(f'[TEST - {args.method.upper()}] {metric_name} over Checkpoints')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = metric_name.lower().replace("-", "_")
        plt.savefig(os.path.join(args.log_dir, f'test_{args.method}_{safe_name}.png'), dpi=150)
        plt.close()

    def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name):
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_mic, 'b-o', label=f'Micro-{category_name}', linewidth=1.5, markersize=3)
        plt.plot(x_vals, y_mac, 'g-s', label=f'Macro-{category_name}', linewidth=1.5, markersize=3)
        plt.plot(x_vals, y_wei, 'r-^', label=f'Weighted-{category_name}', linewidth=1.5, markersize=3)
        plt.xlabel('Global Round / Checkpoint Index')
        plt.ylabel(f'{category_name} (%)')
        plt.title(f'[TEST - {args.method.upper()}] {category_name} (Micro vs Macro vs Weighted)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = category_name.lower().replace("-", "_")
        plt.savefig(os.path.join(args.log_dir, f'test_{args.method}_combined_{safe_name}.png'), dpi=150)
        plt.close()

    if accuracy_history:
        x_axis = global_round_history
        # Bieu do don
        save_test_plot(x_axis, accuracy_history, 'Accuracy', 'b', 'o')
        save_test_plot(x_axis, loss_history, 'Loss', 'k', 'X')
        
        # Bieu do ket hop (Micro, Macro, Weighted)
        save_combined_plot(x_axis, prec_mic_history, prec_mac_history, prec_wei_history, 'Precision')
        save_combined_plot(x_axis, rec_mic_history, rec_mac_history, rec_wei_history, 'Recall')
        save_combined_plot(x_axis, f1_mic_history, f1_mac_history, f1_wei_history, 'F1-Score')
        
        logger.info(f'[TEST] Da ve bieu do don va ket hop vao: {args.log_dir}')


def main():
    # === Parse arguments ===
    args = args_parser()
    
    # === Setup ===
    setup_seed(args.seed)
    logger = setup_logging(args.log_dir, args)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info('='*60)
    logger.info('HFIN - Hierarchical Federated Class-Incremental Learning')
    logger.info('   Network Intrusion Detection for IIoT')
    logger.info('='*60)
    logger.info(f'Device: {args.device}')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Method: {args.method.upper()}')
    logger.info(f'Mode: {args.mode.upper()}')
    logger.info(f'Clients: {args.num_clients}, Edge Servers: {args.num_edge_servers}')
    logger.info(f'Base classes: {args.num_base_classes}, Task size: {args.task_size}')
    logger.info(f'Total classes: {args.total_classes}')

    if args.mode == 'test':
        _run_test_mode(args, logger)
        return
    
    # === Load & Tiền xử lý dữ liệu tự động ===
    logger.info(f'\n[1/4] Loading global test data for {args.dataset.upper()}...')
    
    # Cấu hình schedule và clients theo dataset
    from data.fl_dataset_loader import (
        FL_TASK_CLASSES_SEQUENTIAL_UQ, 
        FL_TASK_CLASSES_SEQUENTIAL_TON, 
        FL_TASK_CLASSES_SEQUENTIAL_CIC
    )
    if args.dataset == 'nf_uq_nids':
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_UQ
        args.num_clients = 60  
        args.num_edge_servers = 3
    elif args.dataset == 'nf_ton_iot':
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_TON
        args.num_clients = 60
        args.num_edge_servers = 3
    else:
        curr_schedule = FL_TASK_CLASSES_SEQUENTIAL_CIC
        args.num_clients = 10 # CIC partition has 10 clients

    # Global test data (dùng chung cho mọi task, đánh giá global model)
    X_test, y_test = load_fl_global_test(args.data_path, args.dataset)
    
    num_features = X_test.shape[1]
    # args.total_classes đã được set trong _fill_dataset_defaults, nhưng có thể cập nhật lại từ dữ liệu
    unique_test_labels = torch.unique(y_test)
    if args.dataset == 'cic_iot23':
        args.total_classes = 34
    else:
        args.total_classes = int(unique_test_labels.max().item()) + 1
    
    logger.info(f'Features: {num_features}, Global Test: {len(X_test)}')
    logger.info(f'Total classes: {args.total_classes} | Clients: {args.num_clients} | Edges: {args.num_edge_servers}')
    
    # Test dataset (dùng chung cho đánh giá)
    # Convert back to numpy for NetFlowDataset compatibility if they are tensors
    if torch.is_tensor(X_test):
        X_test = X_test.numpy()
    if torch.is_tensor(y_test):
        y_test = y_test.numpy()
        
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
            wto_beta=args.wto_beta
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

    # ── CSV để lưu mẽtric theo từng round (chuẩn chung SPCIL/MalCL) ───────────
    import csv as _csv
    _csv_path = os.path.join(args.log_dir, f'metrics_{args.method}.csv')
    _csv_file = open(_csv_path, 'w', newline='', encoding='utf-8')
    _csv_writer = _csv.writer(_csv_file)
    _csv_writer.writerow([
        'task', 'round', 'global_round', 'task_progress',
        'acc',
        'prec_mic', 'prec_mac', 'prec_wei',
        'rec_mic',  'rec_mac',  'rec_wei',
        'f1_mic',   'f1_mac',   'f1_wei',
        'fpr', 'loss',
    ])
    
    task_accuracies_per_class = {}  # Lưu accuracy từng lớp theo round để tính forgetting
    classes_learned = args.num_base_classes
    current_f1_scores = {i: 0.9 for i in range(args.total_classes)}
    global_round = 0  # Bộ đếm round tổng (để log)
    
    start_task_id = 0
    start_round_in_task = 0

    # === Resume Logic ===
    if args.resume and args.resume_ckpt:
        if os.path.isfile(args.resume_ckpt):
            logger.info(f'\n[RESUME] Loading checkpoint: {args.resume_ckpt}')
            checkpoint = torch.load(args.resume_ckpt, map_location=args.device, weights_only=False)
            
            # Restore metadata
            start_task_id = checkpoint.get('task_id', 0)
            start_round_in_task = checkpoint.get('round_in_task', 0) # checkpoint saves round_in_task+1
            global_round = checkpoint.get('global_round', 0)
            classes_learned = checkpoint.get('classes_learned', args.num_base_classes)
            
            # Expand model to match checkpoint classes
            if classes_learned > model_g.out_features:
                if isinstance(model_g, DERNetwork):
                    # For DER, we need to know how many backbones to add. 
                    # Use actual task sizes from curr_schedule instead of constant args.task_size
                    curr_cls = len(curr_schedule[0])
                    for t in range(1, start_task_id + 1):
                         curr_cls += len(curr_schedule[t])
                         model_g.update_fc(curr_cls)
                else:
                    model_g.Incremental_learning(classes_learned)
                logger.info(f'[RESUME] Expanded model to {classes_learned} classes')
            
            # Load weights
            model_g.load_state_dict(checkpoint['model_state_dict'])
            cloud_server.model = copy.deepcopy(model_g)
            for edge in edge_servers:
                edge.model = copy.deepcopy(model_g)
                edge.learned_classes = list(range(classes_learned))
            
            # Restore Edge Server Memory if available
            if 'edge_memories' in checkpoint:
                logger.info('[RESUME] Restoring Edge Server memories...')
                for e_id, mem in enumerate(checkpoint['edge_memories']):
                    if e_id < len(edge_servers):
                        edge_servers[e_id].exemplar_manager.exemplar_set = mem['exemplar_set']
                        edge_servers[e_id].exemplar_manager.exemplar_labels = mem['exemplar_labels']
            
            logger.info(f'[RESUME] Resuming from Task {start_task_id}, Round {start_round_in_task} (Global Round {global_round})')
        else:
            logger.error(f'[RESUME] Checkpoint file not found: {args.resume_ckpt}')
            return

    # Phân chia Task tự động theo bài báo
    all_task_classes = [curr_schedule[i] for i in range(len(curr_schedule))]
    num_tasks = len(all_task_classes)
    
    # Label map không còn dùng nữa, tạo label map giả định (0 -> Class 0)
    label_map = {f'Class {i}': i for i in range(args.total_classes)}
    # Đảo ngược label_map để in report đẹp hơn
    inv_label_map = {v: k for k, v in label_map.items()}
    # Nếu label_map có string keys, ta sẽ dùng trực tiếp cho report

    for task_id, task_classes in enumerate(all_task_classes):
        if args.resume and task_id < start_task_id:
            logger.info(f'Skipping Task {task_id} (already completed)')
            continue
            
        logger.info(f'\nLoading data for Task {task_id} (Classes: {task_classes})...')
        
        # ── 1. Chuẩn bị dữ liệu cho Task hiện tại ────────────────────────
        update_clients_for_task(clients_dict, args.data_path, task_id, args.dataset)
        
        # Rút gọn dữ liệu cho debug mode
        if args.debug and args.max_samples > 0:
            for cid, client in clients_dict.items():
                if len(client.train_data) > args.max_samples:
                    client.train_data = client.train_data[:args.max_samples]
                    client.train_labels = client.train_labels[:args.max_samples]
        
        # Số rounds cho task này (Sec VI.B)
        num_rounds = args.epochs_base if task_id == 0 else args.epochs_incremental

        # Cập nhật số lớp đã học dựa trên cấu hình task thực tế (an toàn khi Resume)
        classes_learned = sum(len(curr_schedule[i]) for i in range(task_id + 1))

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
            if args.resume and task_id == start_task_id and round_in_task < start_round_in_task:
                continue
            
            global_round += 1

            logger.info(f'\n--- Task {task_id}, Round {round_in_task + 1}/{num_rounds} '
                        f'(Global {global_round}) ---')

            # --- LR Decay logic (Paper Sec VI.B) ---
            # Milestones: 40, 60 for 80 rounds; 20, 30 for 40 rounds
            milestones = [20, 30] if task_id == 0 else [40, 60]
            if round_in_task in milestones:
                current_lr *= args.lr_decay_gamma
                logger.info(f'   [LR DECAY] Round {round_in_task} reached milestone: New LR = {current_lr}')

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
            cloud_server.model = copy.deepcopy(model_g)

            if is_first_round or is_eval_round or is_last_round_of_task:
                results_eval = evaluate_model(model_g, test_dataset, range(classes_learned), args.device)
                acc = results_eval['accuracy']
                precision_mac = results_eval['precision_macro']
                recall_mac = results_eval['recall_macro']
                f1_mic = results_eval['f1_micro']
                f1_macro = results_eval['f1_macro']
                f1_weighted = results_eval['f1_weighted']
                loss = results_eval['loss']
                
                accuracy_history.append(acc)
                precision_history.append(precision_mac)
                recall_history.append(recall_mac)
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
                    f'Prec-Mac: {precision_mac:.2f}% | '
                    f'Rec-Mac: {recall_mac:.2f}% | '
                    f'Mi-F1: {f1_mic:.2f}% | '
                    f'Ma-F1: {f1_macro:.2f}% | '
                    f'We-F1: {f1_weighted:.2f}% | '
                    f'Loss: {loss:.4f}'
                )

                # ── Ghi vào metrics.csv sau mỗi lần eval ──────────────────────────
                _csv_writer.writerow([
                    task_id, round_in_task + 1, global_round,
                    round(task_progress, 4),
                    round(acc, 4),
                    round(results_eval.get('precision_micro', 0), 4),
                    round(precision_mac, 4),
                    round(results_eval.get('precision_weighted', 0), 4),
                    round(results_eval.get('recall_micro', 0), 4),
                    round(recall_mac, 4),
                    round(results_eval.get('recall_weighted', 0), 4),
                    round(f1_mic, 4),
                    round(f1_macro, 4),
                    round(f1_weighted, 4),
                    round(results_eval.get('fpr', 0), 4),
                    round(loss, 6),
                ])
                _csv_file.flush()  # dam bao ghi xuong dia ngay

                # ── Luu checkpoint sau moi eval round ───────────────────────────
                ckpt_filename = (
                    f'ckpt_{args.method}_task{task_id:02d}_r{round_in_task+1:03d}'
                    f'_global{global_round:03d}_acc{acc:.1f}.pth'
                )
                ckpt_path = os.path.join(args.checkpoint_dir, ckpt_filename)
                # --- Collect Edge Server memories for full checkpoint ---
                edge_memories = []
                for edge in edge_servers:
                    edge_memories.append({
                        'exemplar_set': edge.exemplar_manager.exemplar_set,
                        'exemplar_labels': edge.exemplar_manager.exemplar_labels
                    })

                torch.save({
                    'task_id':       task_id,
                    'round_in_task': round_in_task + 1,
                    'global_round':  global_round,
                    'method':        args.method,
                    'classes_learned': classes_learned,
                    'model_state_dict': model_g.state_dict(),
                    'edge_memories': edge_memories, # Important for full resume
                    'metrics': {
                        'accuracy':           acc,
                        'precision_micro':    results_eval.get('precision_micro', 0),
                        'precision_macro':    precision_mac,
                        'precision_weighted': results_eval.get('precision_weighted', 0),
                        'recall_micro':       results_eval.get('recall_micro', 0),
                        'recall_macro':       recall_mac,
                        'recall_weighted':    results_eval.get('recall_weighted', 0),
                        'f1_micro':           f1_mic,
                        'f1_macro':           f1_macro,
                        'f1_weighted':        f1_weighted,
                        'loss':               loss,
                    },
                }, ckpt_path)
                logger.info(f'  [CKPT] Saved: {ckpt_filename}')

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
            results_post = evaluate_model(model_g, test_dataset, range(classes_learned), args.device)
            acc_post = results_post['accuracy']
            # Cập nhật lại cloud server với các trọng số đã được align
            cloud_server.model = copy.deepcopy(model_g)
            
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

    # Đóng file CSV metrics
    _csv_file.close()
    logger.info(f'Metrics CSV saved: {_csv_path}')


if __name__ == '__main__':
    main()
