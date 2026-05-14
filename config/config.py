"""
Cau hinh va tham so cho HFIN - Hierarchical Federated Incremental Learning NID
Thong so duoc can chinh theo Muc VI.B cua bai bao.
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser(
        description='HFIN - Hierarchical Federated Class-Incremental Learning for NID'
    )

    # === Dataset ===
    parser.add_argument('--dataset', type=str, default='cic_iot23',
                        choices=['nf_ton_iot', 'nf_uq_nids', 'nf_unsw_nb15', 'cic_iot23'],
                        help='Ten dataset (mac dinh: cic_iot23)')
    parser.add_argument('--data_path', type=str, default=r'c:\FederatedLearning\FL\core\data_split',
                        help='Duong dan thu muc raw chua file CSV va .pkl')
    parser.add_argument('--num_features', type=int, default=46,
                        help='So features dau vao (CIC-IoT23: 46)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Gioi han mau (0 = lay het)')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Ti le test set (0.3 = 70/30)')

    # === Task Schedule (Class-Incremental Learning) ===
    parser.add_argument('--task_schedule', type=str, default='task6',
                        choices=['task2', 'task4', 'task5', 'task10', 'task6'],
                        help='Cau hinh task incremental')
    parser.add_argument('--num_base_classes', type=int, default=6,
                        help='So lop base task')
    parser.add_argument('--task_size', type=int, default=6,
                        help='So lop moi moi incremental task')
    parser.add_argument('--total_classes', type=int, default=34,
                        help='Tong so class: cic_iot23=34')
    parser.add_argument('--memory_size', type=int, default=500,
                        help='Kich thuoc bo nho exemplar tai Edge Server')

    # === Federated Learning ===
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Tong so clients')
    parser.add_argument('--num_edge_servers', type=int, default=2,
                        help='So edge servers')
    parser.add_argument('--local_clients', type=int, default=10,
                        help='So clients duoc chon moi vong')
    parser.add_argument('--epochs_base', type=int, default=40,
                        help='So global rounds cho Base Task')
    parser.add_argument('--epochs_incremental', type=int, default=80,
                        help='So global rounds cho moi Incremental Task')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Tham so Dirichlet non-IID')
    parser.add_argument('--alpha_benign', type=float, default=0.3,
                        help='Alpha cho lop Benign')
    parser.add_argument('--alpha_attack', type=float, default=0.8,
                        help='Alpha cho cac lop Attack')

    # === Huan luyen ===
    parser.add_argument('--epochs_local', type=int, default=1,
                        help='So epochs huan luyen local moi round')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate base task')
    parser.add_argument('--lr_incremental', type=float, default=0.02,
                        help='Learning rate incremental task')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD Momentum')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='Giam LR sau moi N global rounds')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                        help='He so giam LR')

    # === Downsampling ===
    parser.add_argument('--max_samples_per_class', type=int, default=0,
                        help='Gioi han so mau toi da moi lop. 0 = khong gioi han.')
    parser.add_argument('--downsample_ratio', type=float, default=1.0,
                        help='Ti le lay mau (1.0 = lay het, 0.125 = 1/8)')

    # === WTO - Weighted Transmission Optimization ===
    parser.add_argument('--wto_beta', type=float, default=0.5,
                        help='Beta trong WTO (Eq. 8)')
    parser.add_argument('--max_transmission_time', type=float, default=2.0,
                        help='Gioi han thoi gian truyen tai WTO (giay)')

    # === Knowledge Distillation ===
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature distillation (T=2)')

    # === Method (Incremental Learning Strategy) ===
    parser.add_argument('--method', type=str, default='wa',
                        choices=['icarl', 'wa', 'der', 'der++'],
                        help='Chien luoc chong catastrophic forgetting: icarl / wa / der / der++')
    parser.add_argument('--der_alpha', type=float, default=0.5,
                        help='DER: trong so MSE term')
    parser.add_argument('--der_beta', type=float, default=0.5,
                        help='DER++: trong so CE-buffer term')

    # === Model ===
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Chieu output cua CNN feature extractor')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # === Khac ===
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='So global rounds giua moi lan danh gia')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Thu muc luu log')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Thu muc luu model (cu, giu lai de tuong thuoc)')
    parser.add_argument('--debug', action='store_true',
                        help='Che do debug: dung max_samples=50000 va epochs_global=2')

    # === Mode: train or test ===
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help=(
                            'train: chay toan bo training va luu checkpoint moi round. '
                            'test : chi load checkpoint da luu va chay evaluation.'
                        ))
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Thu muc de luu checkpoint trong qua trinh train.')
    parser.add_argument('--test_checkpoint_dir', type=str, default='',
                        help=(
                            'Thu muc chua checkpoint de load khi chay --mode test. '
                            'Neu de trong, se tim thu muc moi nhat trong checkpoint_dir.'
                        ))
    
    # === Resume Training ===
    parser.add_argument('--resume', action='store_true',
                        help='Bat che do resume tu checkpoint.')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='Duong dan den file .pth de resume.')

    args = parser.parse_args()

    # === Auto-fill tu dataset & task_schedule ===
    _fill_dataset_defaults(args)

    # Auto detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Debug mode
    if args.debug:
        if args.max_samples == 0:
            args.max_samples = 50000
        args.epochs_base        = 2
        args.epochs_incremental = 2
        args.num_clients        = 10
        args.num_edge_servers   = 2
        args.local_clients      = 10
        args.epochs_local       = 1

    return args


def _fill_dataset_defaults(args):
    """Tự động điền thông số dựa trên dataset được chọn."""
    if args.dataset == 'nf_uq_nids':
        args.total_classes    = 21
        args.num_base_classes = 5
        args.task_size        = 4
        args.num_features     = 39
        args.num_clients      = 60
        args.num_edge_servers = 3
        if args.data_path == r'c:\FederatedLearning\FL\core\data_split':
             args.data_path = r'D:\IDPS\HFIN\IDPS\data\raw'
    elif args.dataset == 'nf_ton_iot':
        args.total_classes    = 10
        args.num_base_classes = 2
        args.task_size        = 2
        args.num_features     = 39
        args.num_clients      = 60
        args.num_edge_servers = 3
        if args.data_path == r'c:\FederatedLearning\FL\core\data_split':
             args.data_path = r'D:\IDPS\HFIN\IDPS\data\raw'
    else:
        # Mặc định cho CIC-IoT23
        args.total_classes    = 34
        args.num_base_classes = 6
        args.task_size        = 6
        args.num_features     = 46
        args.dataset          = 'cic_iot23'

    print(f'[CONFIG] Dataset: {args.dataset}')
    print(f'         Total classes: {args.total_classes} | '
          f'Features: {args.num_features} | '
          f'Base: {args.num_base_classes} | '
          f'Step: {args.task_size} class/task')
