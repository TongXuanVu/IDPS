"""
Cấu hình và tham số cho HFIN - Hierarchical Federated Incremental Learning NID
Thông số được căn chỉnh theo Mục VI.B của bài báo.
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(
        description='HFIN - Hierarchical Federated Class-Incremental Learning for NID'
    )

    # === Dataset ===
    parser.add_argument('--dataset', type=str, default='nf_ton_iot',
                        choices=['nf_ton_iot', 'nf_uq_nids', 'nf_unsw_nb15', 'cic_iot23'],
                        help='Ten dataset (mac dinh: nf_ton_iot)')
    parser.add_argument('--data_path', type=str, default=r'd:\IDPS\HFIN\IDPS\data\raw',
                        help='Duong dan thu muc raw chua file CSV va .pkl (mac dinh: data/raw/)')
    parser.add_argument('--num_features', type=int, default=46,
                        help='Số features đầu vào (CIC-IoT23 mới: 46, cũ: 32)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Giới hạn mẫu (0 = lấy hết). Không cần thiết vì data chia ra từ trước.')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Tỷ lệ test set (0.3 = 70/30 theo yêu cầu FL)')

    # === Task Schedule (Class-Incremental Learning) ===
    # Tham khảo Table I trong bài báo
    parser.add_argument('--task_schedule', type=str, default='task2',
                        choices=['task2', 'task4', 'task5', 'task10', 'task6'],
                        help=(
                            'Cau hinh task incremental: '
                            'nf_ton_iot: task2 (5 tasks +2 class) hoac task5 (2 tasks +5 class); '
                            'nf_uq_nids: task2/task4/task10; '
                            'cic_iot23: task6'
                        ))
    parser.add_argument('--num_base_classes', type=int, default=6,
                        help='Số lớp base task (tự động ghi đè từ task_schedule nếu 0)')
    parser.add_argument('--task_size', type=int, default=6,
                        help='Số lớp mới mỗi incremental task (ghi đè từ schedule)')
    parser.add_argument('--total_classes', type=int, default=34,
                        help='Tổng số class: nf_ton_iot=10, nf_uq_nids=21, cic_iot23=34')
    parser.add_argument('--memory_size', type=int, default=2000,
                        help='Kích thước bộ nhớ exemplar tại Edge Server (Bài báo: 1000-2000)')

    # === Federated Learning ===
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Tổng số clients (FL partition mới: 10 clients)')
    parser.add_argument('--num_edge_servers', type=int, default=2,
                        help='Số edge servers: Bài báo đề xuất cấu trúc phân cấp với 2 Edge Servers')
    parser.add_argument('--local_clients', type=int, default=10,
                        help='Số clients được chọn mỗi vòng (bằng num_clients)')
    parser.add_argument('--epochs_base', type=int, default=30,
                        help='Số global rounds cho Base Task')
    parser.add_argument('--epochs_incremental', type=int, default=30,
                        help='Số global rounds cho mỗi Incremental Task')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Tham số Dirichlet non-IID (nhỏ → non-IID mạnh hơn)')
    parser.add_argument('--alpha_benign', type=float, default=0.3,
                        help='Alpha cho lớp Benign (0.3 theo Mục VI.B)')
    parser.add_argument('--alpha_attack', type=float, default=0.8,
                        help='Alpha cho các lớp Attack (0.8 theo Mục VI.B)')

    # === Huấn luyện ===
    parser.add_argument('--epochs_local', type=int, default=5,
                        help='Số epochs huấn luyện local mỗi round (bài báo dùng 2 đến 5)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (Paper Table I: 1024)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate base task (Mục VI.B: 1e-2, user đề xuất: 1e-3)')
    parser.add_argument('--lr_incremental', type=float, default=0.001,
                        help='Learning rate incremental task (Mục VI.B: 2e-2, user đề xuất: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (Mục VI.B: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD Momentum (Mục VI.B: 0.9)')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Giảm LR sau mỗi N global rounds')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                        help='Hệ số giảm LR')

    # === Downsampling (Paper Section VI.B) ===
    parser.add_argument('--max_samples_per_class', type=int, default=0,
                        help='Giới hạn số mẫu tối đa mỗi lớp. 0 = không dùng giới hạn cứng.')
    parser.add_argument('--downsample_ratio', type=float, default=0.125,
                        help=(
                            'Tỷ lệ lấy mẫu (ví dụ: 0.125 là 1/8). '
                            'Chỉ áp dụng nếu max_samples_per_class = 0. '
                            'Paper NF-UQ áp dụng 1/8 cho class 1-3 và 1/3 cho class 4-8.'
                        ))

    # === WTO - Weighted Transmission Optimization ===
    parser.add_argument('--wto_beta', type=float, default=0.5,
                        help='Beta trong WTO (Eq. 8): cân bằng class importance')
    parser.add_argument('--max_transmission_time', type=float, default=2.0,
                        help='Giới hạn thời gian truyền tải WTO (giây)')

    # === Knowledge Distillation ===
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature distillation (T=2, Mục IV.B)')

    # === Method (Incremental Learning Strategy) ===
    parser.add_argument('--method', type=str, default='wa',
                        choices=['icarl', 'wa', 'der', 'der++'],
                        help=(
                            'Chiến lược chống catastrophic forgetting:\n'
                            '  icarl : iCaRL thuần (chỉ KD + Exemplar)\n'
                            '  wa    : iCaRL + Weight Aligning (bài báo gốc HFIN)\n'
                            '  der   : Dark Experience Replay (Buzzega 2020)\n'
                            '  der++ : DER++ (thêm CE trên buffer samples)'
                        ))
    parser.add_argument('--der_alpha', type=float, default=0.5,
                        help='DER: trọng số MSE term (default 0.5)')
    parser.add_argument('--der_beta', type=float, default=0.5,
                        help='DER++: trọng số CE-buffer term (default 0.5)')

    # === Model ===
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Chiều output của CNN feature extractor')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # === Khác ===
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Số global rounds giữa mỗi lần đánh giá (1 = đánh giá mỗi round)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Thư mục lưu log')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Thư mục lưu model checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Chế độ debug: dùng max_samples=50000 và epochs_global=2')

    args = parser.parse_args()

    # === Auto-fill từ dataset & task_schedule ===
    _fill_dataset_defaults(args)

    # Auto detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Debug mode: giảm số mẫu và vòng lặp
    if args.debug:
        if args.max_samples == 0:
            args.max_samples = 50000
        args.epochs_base        = 2
        args.epochs_incremental = 2
        args.num_clients        = 10
        args.num_edge_servers   = 2   # 2 edge servers theo bai bao
        args.local_clients      = 10
        args.epochs_local       = 1

    return args


def _fill_dataset_defaults(args):
    """Tự động điền total_classes, num_base_classes, task_size từ dataset & schedule."""
    from data.partition import TASK_CONFIGS

    if args.dataset not in TASK_CONFIGS:
        return

    cfg   = TASK_CONFIGS[args.dataset]
    key   = args.task_schedule

    # Lấy schedule khả dụng
    avail = list(cfg['schedules'].keys())
    if key not in avail:
        key = cfg['default_schedule']
        # Dung ASCII de tranh UnicodeEncodeError tren Windows console
        print(f'[CONFIG] task_schedule "{args.task_schedule}" khong kha dung voi dataset '
              f'"{args.dataset}". Dung "{key}" thay the. Kha dung: {avail}')
        args.task_schedule = key

    sched = cfg['schedules'][key]
    args.total_classes    = cfg['total_classes']
    args.num_base_classes = sched['base']
    args.task_size        = sched['step']

    print(f'[CONFIG] Dataset: {args.dataset} | Schedule: {key}')
    print(f'         Total classes: {args.total_classes} | '
          f'Base: {args.num_base_classes} | '
          f'Step: {args.task_size} class/task | '
          f'Tasks: {sched["num_tasks"]} incremental tasks')
