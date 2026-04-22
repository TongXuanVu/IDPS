"""
Knowledge Distillation Loss cho Class-Incremental Learning
Giúp model không quên kiến thức cũ khi học lớp mới
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_one_hot(target, num_classes, device):
    """Chuyển label thành one-hot vector"""
    one_hot = torch.zeros(target.shape[0], num_classes).to(device)
    one_hot.scatter_(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def distillation_loss(outputs, old_outputs, targets, num_classes,
                      old_num_classes, device, temperature=2.0, is_old_mask=None,
                      weights=None):
    """
    Tính loss kết hợp theo Eq. 7 từ bài báo HFIN, tối ưu hóa Selective KD.
    """
    # 1. Classification loss (cross-entropy) áp dụng cho TOÀN BỘ batch
    # Thêm trọng số nếu được cung cấp để cứu các lớp hiếm
    loss_ce = F.cross_entropy(outputs, targets, weight=weights)

    if old_outputs is None or old_num_classes == 0:
        return loss_ce

    # 2. Knowledge distillation loss (Selective KD)
    # Lọc chỉ lấy các mẫu thuộc lớp cũ nếu có mask
    if is_old_mask is not None and is_old_mask.sum() > 0:
        p_old = F.softmax(old_outputs[is_old_mask] / temperature, dim=1)
        p_new = F.log_softmax(outputs[is_old_mask, :old_num_classes] / temperature, dim=1)
        # Nếu chỉ có ít mẫu cũ trong batch, KD loss sẽ bị nhỏ, cần scale lại
        loss_kd = F.kl_div(p_new, p_old.detach(), reduction='batchmean') * (temperature ** 2)
    elif is_old_mask is None:
        # Fallback về tính toàn bộ batch nếu không có mask
        p_old = F.softmax(old_outputs / temperature, dim=1)
        p_new = F.log_softmax(outputs[:, :old_num_classes] / temperature, dim=1)
        loss_kd = F.kl_div(p_new, p_old.detach(), reduction='batchmean') * (temperature ** 2)
    else:
        # Không có mẫu cũ nào trong batch
        loss_kd = 0.0

    # 3. Hệ số Lambdas (Theo bài báo HFIN lambda1=1.0, lambda2=1.0)
    lambda1 = 1.0
    lambda2 = 1.0
    
    total_loss = lambda1 * loss_ce + lambda2 * loss_kd
    
    return total_loss


def efficient_old_class_weight(output, label, num_classes, learned_classes, device):
    """
    Tính trọng số adaptive cho lớp cũ vs lớp mới
    Tương tự hàm trong GLFC gốc, đảm bảo cân bằng gradient
    
    Args:
        output: logits (batch, num_classes)
        label: nhãn (batch,)
        num_classes: tổng số lớp hiện tại
        learned_classes: list lớp đã học trước đó
        device: str
    
    Returns:
        w: trọng số (batch, 1)
    """
    pred = torch.sigmoid(output)
    N, C = pred.size(0), pred.size(1)
    
    class_mask = pred.data.new(N, C).fill_(0)
    ids = label.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    
    target = get_one_hot(label, num_classes, device)
    g = torch.abs(pred.detach() - target)
    g = (g * class_mask).sum(1).view(-1, 1)
    
    if len(learned_classes) != 0:
        ids_clone = ids.clone()
        for c in learned_classes:
            ids_clone = torch.where(ids_clone != c, ids_clone, ids_clone.clone().fill_(-1))
        
        index_old = torch.eq(ids_clone, -1).float()
        index_new = torch.ne(ids_clone, -1).float()
        
        if index_old.sum() != 0:
            w_old = torch.div(g * index_old, (g * index_old).sum() / index_old.sum())
        else:
            w_old = g.clone().fill_(0.)
        
        if index_new.sum() != 0:
            w_new = torch.div(g * index_new, (g * index_new).sum() / index_new.sum())
        else:
            w_new = g.clone().fill_(0.)
        
        w = w_old + w_new
    else:
        w = g.clone().fill_(1.)
    
    return w
