import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from federated.fed_utils import FedAvg, model_to_device
from incremental.wto import wto_select_clients_for_data
from incremental.exemplar import ExemplarManager
from incremental.distillation import distillation_loss
from models.der_network import DERNetwork
from data.dataset import NetFlowDataset
from models.network import HFINNetwork

logger = logging.getLogger(__name__)


class EdgeServer:
    """
    Edge Server trong kiến trúc phân cấp HFIN.
    Hỗ trợ hai chế độ chống catastrophic forgetting:
      - 'icarl' : iCaRL + Knowledge Distillation + Weight Aligning (bài báo gốc)
      - 'der'   : Dark Experience Replay (Buzzega et al., NeurIPS 2020)
      - 'der++' : DER++ (thêm CE term trên buffer samples)
    """

    def __init__(self, edge_id, num_classes, feature_extractor,
                 device='cpu', memory_size=500, task_size=2,
                 method='icarl', der_alpha=0.5, der_beta=0.5,
                 max_samples_per_class=0, downsample_ratio=0.125,
                 input_dim=41, feature_dim=64, lambda_aux=1.0):
        """
        Args:
            edge_id               : ID cua edge server
            num_classes           : So lop ban dau
            feature_extractor     : Backbone CNN (dung cho iCaRL/WA)
            device                : str ('cpu' | 'cuda')
            memory_size           : Tong so mau trong memory (iCaRL)
            task_size             : So lop moi moi task
            method                : 'icarl' | 'wa' | 'der'
            der_alpha             : (khong dung voi DER dynamic, giu lai de tuong thich)
            der_beta              : (khong dung voi DER dynamic, giu lai de tuong thich)
            max_samples_per_class : Gioi han mau/lop. 0 = khong gioi han.
            input_dim             : So features dau vao (cho DERNetwork)
            feature_dim           : Output dim moi CNN backbone (cho DERNetwork)
            lambda_aux            : Trong so loss aux cua DER (mac dinh 1.0 theo SPCIL)
        """
        self.edge_id  = edge_id
        self.device   = device
        self.task_size = task_size
        self.client_ids = []
        self.method   = method
        self.der_alpha = der_alpha
        self.der_beta  = der_beta
        self.max_samples_per_class = max_samples_per_class
        self.downsample_ratio = downsample_ratio
        self.lambda_aux = lambda_aux
        self.input_dim  = input_dim
        self.feature_dim = feature_dim

        # === Khoi tao Model theo method ===
        if method in ('der', 'der++'):
            # DER: Dynamic Expansion Network (port tu SPCIL)
            self.model = DERNetwork(input_dim=input_dim, feature_dim=feature_dim)
            # Khoi tao backbone cho base task
            self.model.update_fc(num_classes)
            # DER van can ExemplarManager de chong quen head classifier
            self.exemplar_manager = ExemplarManager(memory_size, feature_dim)
        else:
            # iCaRL / WA: dung HFINNetwork (1 backbone + expandable fc)
            self.model = HFINNetwork(num_classes, feature_extractor)
            feature_ext_dim = feature_extractor.fc.in_features if hasattr(feature_extractor, 'fc') else 64
            self.exemplar_manager = ExemplarManager(memory_size, feature_ext_dim)

        # Training state
        self.old_model   = None
        self.learned_classes = []
        self.task_id_old = -1

    # ------------------------------------------------------------------
    def set_clients(self, client_ids):
        """Gán danh sách clients thuộc phạm vi quản lý của Edge này"""
        self.client_ids = client_ids

    # ------------------------------------------------------------------
    def train_local(self, clients_dict, global_round, task_id,
                    task_classes, current_f1_scores,
                    epochs=5, lr=0.01, batch_size=32, is_last_round=False):
        """
        Huấn luyện cục bộ tại Edge Server.

        Args:
            clients_dict     : Dict {client_id: HFINClient}
            global_round     : Round hiện tại (toàn cục)
            task_id          : ID task (0-indexed)
            task_classes     : List các class thuộc task hiện tại
            current_f1_scores: Dict {class_id: f1} từ round trước (WTO)
            epochs           : Số epoch local training
            lr               : Learning rate
            batch_size       : Batch size
            is_last_round    : True nếu là round cuối của task
                               → Cập nhật exemplar memory (chỉ iCaRL)
        """
        # ── 1. WTO: Chọn clients ──────────────────────────────────────
        client_infos = []
        for cid in self.client_ids:
            client = clients_dict[cid]
            client_infos.append({
                'client_id': cid,
                'class_counts': client.get_class_counts(),
                'transmission_rate': 10.0
            })

        selected_client_ids = wto_select_clients_for_data(
            client_infos, current_f1_scores, beta=0.5
        )

        if not selected_client_ids:
            logger.info(f"Edge {self.edge_id}: WTO không chọn client nào.")
            return self.get_weights(), 0

        # ── 2. Thu thập dữ liệu từ selected clients ───────────────────
        all_data, all_labels = [], []
        for cid in selected_client_ids:
            data, labels = clients_dict[cid].get_data_for_edge(task_classes)
            if data is not None and len(data) > 0:
                all_data.append(data)
                all_labels.append(labels)

        if not all_data:
            logger.warning(f"Edge {self.edge_id}: Không có dữ liệu từ bất kỳ client nào cho task này.")
            return self.get_weights(), 0

        X_train = torch.cat(all_data)
        y_train = torch.cat(all_labels)

        # ── 2b. Per-class Downsampling (Paper Section VI.B) ──────────────────────
        # Áp dụng strict cap (max_samples_per_class) HOẶC ratio (downsample_ratio).
        if self.max_samples_per_class > 0 or (self.downsample_ratio > 0 and self.downsample_ratio < 1.0):
            keep_idx = []
            for cls in torch.unique(y_train):
                cls_mask = (y_train == cls).nonzero(as_tuple=True)[0]
                if self.max_samples_per_class > 0:
                    n_keep = min(len(cls_mask), self.max_samples_per_class)
                else:
                    n_keep = max(1, int(len(cls_mask) * self.downsample_ratio))
                
                perm = torch.randperm(len(cls_mask))[:n_keep]
                keep_idx.append(cls_mask[perm])
            keep_idx = torch.cat(keep_idx)
            keep_idx = keep_idx[torch.randperm(len(keep_idx))]  # shuffle
            X_train = X_train[keep_idx]
            y_train = y_train[keep_idx]
            
            if self.max_samples_per_class > 0:
                msg = f"max {self.max_samples_per_class}/class"
            else:
                msg = f"ratio {self.downsample_ratio}"
                
            logger.info(
                f"Edge {self.edge_id}: Downsampled to "
                f"{len(y_train):,} samples ({msg})."

            )

        # ── 3. Intra-task oversampling (cứu nhãn hiếm trong task hiện tại) ──
        unique_new, counts_new = torch.unique(y_train, return_counts=True)
        min_samples, max_repeat = 30, 10
        oversampled_x = [X_train]
        oversampled_y = [y_train]
        for cls, count in zip(unique_new, counts_new):
            if count < min_samples:
                idx = (y_train == cls)
                rep = min(int(np.ceil(min_samples / count.item())), max_repeat)
                if rep > 1:
                    oversampled_x.append(X_train[idx].repeat(rep - 1, 1))
                    oversampled_y.append(y_train[idx].repeat(rep - 1))
        X_train = torch.cat(oversampled_x)
        y_train = torch.cat(oversampled_y)
        num_new_samples = len(X_train)   # Lưu lại trước khi concat old data

        # ── 4. Task detection (1 lần đầu mỗi task) ────────────────────
        if task_id > self.task_id_old:
            self.task_id_old = task_id
            old_num_classes = self.model.fc.out_features - len(task_classes)
            self.learned_classes = list(range(old_num_classes))

            if self.method in ('icarl', 'wa'):
                # iCaRL / WA: lưu bản copy old model để KD
                self.old_model = copy.deepcopy(self.model)
                self.old_model.eval()
            # DER: không cần old model (logits đã lưu trong buffer)

        if self.method in ('icarl', 'wa') and self.old_model is not None:
            self.old_model.eval()

        # Kiểm tra label hợp lệ
        if len(y_train) > 0:
            max_label = int(y_train.max().item())
            if max_label >= self.model.fc.out_features:
                logger.warning(
                    f"Edge {self.edge_id}: Label {max_label} >= "
                    f"model out_features {self.model.fc.out_features}."
                )

        # ── 5. Class weights (chống imbalance) ─────────────────────────
        self.model = model_to_device(self.model, self.device)
        self.model.train()

        # Tính toán trọng số trên toàn bộ các class đã học (0 -> current_max)
        num_classes_current = self.model.fc.out_features
        class_weights = torch.ones(num_classes_current).to(self.device)
        unique_y, counts_y = torch.unique(y_train, return_counts=True)
        
        # Trọng số cho các lớp mới (dựa trên phân phối thực tế trong task)
        total_samples = len(y_train)
        num_new_classes = len(unique_y)
        
        for cls, count in zip(unique_y, counts_y):
            if cls < num_classes_current:
                # Công thức căn bậc hai để làm dịu trọng số (Square Root Smoothing)
                raw_w = total_samples / (num_new_classes * count.float())
                class_weights[cls] = torch.pow(raw_w, 0.5)
        
        # Đối với các lớp cũ (không có trong y_train nhưng có trong Buffer):
        # Ta giữ trọng số mặc định là 1.0 hoặc có thể boosting nhẹ nếu cần.
        # Ở đây ta chuẩn hóa để giá trị trung bình là 1.0
        class_weights = class_weights / class_weights.mean()

        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=1e-5)

        # ══════════════════════════════════════════════════════════════
        # NHÁNH TRAINING TÙY THEO METHOD
        # ══════════════════════════════════════════════════════════════
        if self.method in ('der', 'der++'):
            # ── DER Dynamic Expansion Training (port tu SPCIL DERNet) ────
            # Buoc 1: Dong bang backbone cu, chi train backbone moi nhat + fc + aux_fc
            if task_id > 0:
                self.model.freeze_old_backbones()

            # Buoc 2: Them du lieu Exemplar tu cac task cu (chong quen classifier)
            exemplar_data, exemplar_labels = self.exemplar_manager.get_exemplar_data()
            if exemplar_data:
                flat_ex_x = torch.FloatTensor(
                    np.concatenate([np.array(s) for s in exemplar_data])
                )
                flat_ex_y = torch.LongTensor(
                    np.concatenate([
                        np.full(len(s), exemplar_labels[i])
                        for i, s in enumerate(exemplar_data)
                    ])
                )
                
                # Cân bằng (balanced replay) giong iCaRL
                num_new = len(X_train)
                num_old = len(flat_ex_x)
                if num_old > 0 and num_new > num_old:
                    target_old = int(num_new * 0.20 / 0.80)
                    rep = min(max(1, target_old // num_old), 50)
                    if rep > 1:
                        flat_ex_x = flat_ex_x.repeat(rep, 1)
                        flat_ex_y = flat_ex_y.repeat(rep)
                
                X_train = torch.cat([X_train, flat_ex_x])
                y_train = torch.cat([y_train, flat_ex_y])

            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # So classes truoc task nay (known_classes)
            known_classes = self.model.fc.out_features - len(task_classes)

            for epoch in range(epochs):
                self.model.train()
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    logits, aux_logits = self.model(batch_x)

                    # Loss chinh: CE tren toan bo classes
                    loss_clf = torch.nn.functional.cross_entropy(
                        logits, batch_y, weight=class_weights
                    )

                    # Loss phu (aux): relabel new classes 1..n, old -> 0 (theo SPCIL)
                    aux_targets = batch_y.clone()
                    aux_targets = torch.where(
                        aux_targets - known_classes + 1 > 0,
                        aux_targets - known_classes + 1,
                        torch.zeros_like(aux_targets)
                    )
                    loss_aux = torch.nn.functional.cross_entropy(aux_logits, aux_targets)

                    loss = loss_clf + self.lambda_aux * loss_aux

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Buoc 3: Cap nhat memory (sau khi ket thuc task)
            if is_last_round:
                # Da loai bo weight_align khoi DER vi gay tac dung nguoc tren tabular data
                # (Macro-F1 rớt từ 62% xuống 51% sau khi align)
                

                
                # Cap nhat exemplar memory tuong tu iCaRL
                total_learned = len(self.exemplar_manager.exemplar_set) + len(task_classes)
                m = self.exemplar_manager.memory_size // total_learned
                self.exemplar_manager.reduce_exemplar_sets(m)

                new_x = X_train[:num_new_samples]
                new_y = y_train[:num_new_samples]
                for cls in np.unique(new_y.cpu().numpy()):
                    if cls in task_classes:
                        cls_mask = (new_y == cls)
                        cls_data = new_x[cls_mask].cpu().numpy()
                        self.exemplar_manager.construct_exemplar_set(
                            cls_data, int(cls), self.model, self.device, m=m
                        )

            logger.info(
                f"Edge {self.edge_id}: [DER] Task {task_id} done. "
                f"Backbones: {len(self.model.convnets)} | "
                f"Total features: {self.model.total_feature_dim} | "
                f"Memory: {self.exemplar_manager.total_stored_samples}"
            )

        else:
            # ── iCaRL Training Loop ────────────────────────────────────
            # 5a. Thêm exemplar replay + balanced oversampling
            exemplar_data, exemplar_labels = self.exemplar_manager.get_exemplar_data()
            is_old = [False] * num_new_samples

            if exemplar_data:
                flat_ex_x = torch.FloatTensor(
                    np.concatenate([np.array(s) for s in exemplar_data])
                )
                flat_ex_y = torch.LongTensor(
                    np.concatenate([
                        np.full(len(s), exemplar_labels[i])
                        for i, s in enumerate(exemplar_data)
                    ])
                )
                # Balanced replay: target 20% of batch, cap 50x
                num_new = len(X_train)
                num_old = len(flat_ex_x)
                if num_old > 0 and num_new > num_old:
                    target_old = int(num_new * 0.20 / 0.80)
                    rep = min(max(1, target_old // num_old), 50)
                    if rep > 1:
                        logger.info(
                            f"Edge {self.edge_id}: Balanced replay "
                            f"{num_old}->{num_old*rep} ({rep}x) for {num_new} new samples."
                        )
                        flat_ex_x = flat_ex_x.repeat(rep, 1)
                        flat_ex_y = flat_ex_y.repeat(rep)

                is_old.extend([True] * len(flat_ex_x))
                X_train = torch.cat([X_train, flat_ex_x])
                y_train = torch.cat([y_train, flat_ex_y])

            is_old_mask = torch.BoolTensor(is_old)

            class _DS(torch.utils.data.Dataset):
                def __init__(self, x, y, mask):
                    self.x, self.y, self.m = x, y, mask
                def __len__(self):  return len(self.y)
                def __getitem__(self, i): return self.x[i], self.y[i], self.m[i]

            loader = DataLoader(_DS(X_train, y_train, is_old_mask),
                                batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                for features, targets, mask in loader:
                    features = features.to(self.device)
                    targets  = targets.to(self.device)
                    mask     = mask.to(self.device)

                    outputs = self.model(features)
                    old_outputs = self.old_model(features) if self.old_model else None

                    loss = distillation_loss(
                        outputs, old_outputs, targets,
                        self.model.fc.out_features,
                        self.old_model.fc.out_features if self.old_model else 0,
                        self.device,
                        is_old_mask=mask,
                        weights=class_weights,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 5b. Cập nhật exemplar memory (chỉ cuối task)
            if is_last_round:
                # Thêm Weight Align cho method 'wa'
                if self.method == 'wa' and task_id > 0:
                    old_classes = self.old_model.fc.out_features
                    new_classes = self.model.fc.out_features
                    self.model.weight_align(old_classes, new_classes)
                    logger.info(f"Edge {self.edge_id}: [WA] Weight Align applied ({old_classes} -> {new_classes}).")

                total_learned = len(self.exemplar_manager.exemplar_set) + len(task_classes)
                m = self.exemplar_manager.memory_size // total_learned
                self.exemplar_manager.reduce_exemplar_sets(m)

                new_x = X_train[:num_new_samples]
                new_y = y_train[:num_new_samples]
                for cls in np.unique(new_y.cpu().numpy()):
                    if cls in task_classes:
                        cls_mask = (new_y == cls)
                        cls_data = new_x[cls_mask].cpu().numpy()
                        self.exemplar_manager.construct_exemplar_set(
                            cls_data, int(cls), self.model, self.device, m=m
                        )
                logger.info(
                    f"Edge {self.edge_id}: Memory updated at end of task. "
                    f"Total stored: {self.exemplar_manager.total_stored_samples} samples "
                    f"across {self.exemplar_manager.num_stored_classes} classes."
                )

        return self.get_weights(), len(X_train)

    # ------------------------------------------------------------------
    def get_weights(self):
        """Trả về state_dict của model"""
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, weights):
        """Cập nhật trọng số từ Cloud Server"""
        self.model.load_state_dict(weights)
