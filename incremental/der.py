"""
DER / DER++ Buffer cho Class-Incremental Learning
Dark Experience Replay (Buzzega et al., NeurIPS 2020)

Ý tưởng: Lưu (x, y, logits) — trong đó logits là output của model TẠI THỜI ĐIỂM
mẫu đó được học. Khi replay, buộc model hiện tại phải match các logits đã lưu
→ Chống forgetting mà không cần copy old model hay herding selection.

DER:   L = CE(batch) + α * MSE( f(buf_x), buf_logits )
DER++: L = CE(batch) + α * MSE(...)  + β * CE( f(buf_x), buf_y )
"""
import torch
import numpy as np
import torch.nn.functional as F


class DERBuffer:
    """
    Reservoir Sampling Buffer cho DER / DER++.
    Lưu (x, y, logits) với xác suất đồng đều qua Reservoir Sampling.

    Reservoir Sampling đảm bảo:
        P(sample i trong buffer) = buffer_size / n_seen
    → Phân phối đồng đều qua thời gian, không bias về task mới hay cũ.
    """

    def __init__(self, buffer_size):
        """
        Args:
            buffer_size: Số lượng mẫu tối đa trong buffer
        """
        self.buffer_size = buffer_size
        self.x_buf:      list = []   # raw input tensors
        self.y_buf:      list = []   # true labels
        self.logits_buf: list = []   # soft logits tại thời điểm học
        self.n_seen = 0              # Tổng số mẫu đã thấy (reservoir counter)

    # ------------------------------------------------------------------
    def add_data(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
        """
        Thêm batch dữ liệu vào buffer bằng Reservoir Sampling.

        Args:
            x      : (B, features) — input samples
            y      : (B,)          — true labels
            logits : (B, C)        — model output (trước softmax)
        """
        x      = x.detach().cpu()
        y      = y.detach().cpu()
        logits = logits.detach().cpu()

        for i in range(len(x)):
            self.n_seen += 1
            if len(self.x_buf) < self.buffer_size:
                # Buffer chưa đầy → thêm trực tiếp
                self.x_buf.append(x[i])
                self.y_buf.append(y[i])
                self.logits_buf.append(logits[i])
            else:
                # Reservoir: thay thế mẫu cũ với xác suất buffer_size / n_seen
                j = int(np.random.randint(0, self.n_seen))
                if j < self.buffer_size:
                    self.x_buf[j]      = x[i]
                    self.y_buf[j]      = y[i]
                    self.logits_buf[j] = logits[i]

    # ------------------------------------------------------------------
    def sample(self, n: int):
        """
        Lấy ngẫu nhiên n mẫu từ buffer.

        Xử lý trường hợp model head đã expand:
        - Buffer có thể chứa logits shape [6] (Task 0) và [12] (Task 1)
        - Clip tất cả về min_c (kích thước nhỏ nhất trong batch)
        - Điều này đảm bảo MSE chỉ tính trên "old classes" chung của mọi sample
        - der_loss sẽ clip current_outputs theo c_stored = min_c → đúng chiều

        Returns:
            (x, y, logits) tensors, hoặc (None, None, None) nếu buffer rỗng.
        """
        if len(self.x_buf) == 0:
            return None, None, None

        n = min(n, len(self.x_buf))
        indices = np.random.choice(len(self.x_buf), n, replace=False)

        x = torch.stack([self.x_buf[i] for i in indices])
        y = torch.stack([self.y_buf[i] for i in indices])

        # Lấy logits list, clip về min_c để torch.stack không bị lỗi
        # khi buffer chứa logits từ các task khác nhau (kích thước khác nhau)
        logits_list = [self.logits_buf[i] for i in indices]
        min_c = min(l.shape[0] for l in logits_list)
        logits = torch.stack([l[:min_c] for l in logits_list])

        return x, y, logits

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.x_buf)

    @property
    def is_empty(self) -> bool:
        return len(self.x_buf) == 0


# ======================================================================
# DER / DER++ Loss
# ======================================================================

def der_loss(
    current_outputs:     torch.Tensor,
    ce_targets:          torch.Tensor,
    buf_current_outputs: torch.Tensor | None,
    buf_stored_logits:   torch.Tensor | None,
    buf_labels:          torch.Tensor | None = None,
    alpha:               float = 0.5,
    beta:                float = 0.5,
    weights:             torch.Tensor | None = None,
    der_plus:            bool = True,
) -> torch.Tensor:
    """
    DER / DER++ loss (Buzzega et al., NeurIPS 2020).

    DER:   L = CE(batch) + α * MSE(f(buf_x)[:C_old], buf_logits)
    DER++: L = CE(batch) + α * MSE(...) + β * CE(f(buf_x), buf_y)

    Args:
        current_outputs     : logits batch hiện tại          (B,  C)
        ce_targets          : nhãn batch hiện tại            (B,)
        buf_current_outputs : logits hiện tại trên buf_x     (B', C)
        buf_stored_logits   : logits đã lưu trong buffer     (B', C_old)
                              C_old ≤ C (head có thể đã expand)
        buf_labels          : nhãn thực buffer               (B',) — cho DER++
        alpha               : trọng số DER MSE term
        beta                : trọng số DER++ CE term
        weights             : class weights cho CE của batch hiện tại
        der_plus            : True → DER++, False → DER thuần túy
    """
    # 1. Classification loss trên batch hiện tại
    loss_ce = F.cross_entropy(current_outputs, ce_targets, weight=weights)

    if buf_current_outputs is None or buf_stored_logits is None:
        return loss_ce

    # 2. DER MSE: current logits vs stored logits (chỉ trên old-class outputs)
    # Cần clip theo số chiều của stored_logits (head có thể đã expand)
    c_stored = buf_stored_logits.shape[1]
    buf_current_clipped = buf_current_outputs[:, :c_stored]
    loss_mse = F.mse_loss(buf_current_clipped, buf_stored_logits.detach())

    total_loss = loss_ce + alpha * loss_mse

    # 3. DER++: CE thêm trên buffer samples với ground-truth labels
    if der_plus and buf_labels is not None:
        # Áp dụng trọng số lớp (weights) cho cả phần CE trên Buffer để tối ưu F1
        loss_ce_buf = F.cross_entropy(buf_current_outputs, buf_labels, weight=weights)
        total_loss = total_loss + beta * loss_ce_buf

    return total_loss
