# HFIN: Hierarchical Federated Class-Incremental Learning for NIDS

HFIN (Hierarchical Federated Class-Incremental Learning) là một framework học máy phân tán được thiết kế chuyên biệt cho Hệ thống Phát hiện Xâm nhập Mạng (NIDS) trong môi trường Industrial IoT (IIoT). 

Framework này giải quyết đồng thời hai thách thức lớn: **Bảo mật dữ liệu** (thông qua Federated Learning) và **Thích ứng liên tục** (thông qua Class-Incremental Learning) mà không gây ra hiện tượng quên lãng thảm họa (Catastrophic Forgetting).

---

## 🚀 Các Tính Năng Nổi Bật (Mới Cập Nhật)
Phiên bản mới nhất của HFIN đã nâng cấp thuật toán **DER (Dark Experience Replay / Dynamically Expandable Representation)** đạt mức độ tối ưu cao nhất trên dữ liệu dạng bảng (tabular data), bao gồm:
1. **Dynamic Expansion với Zero-Initialization:** Mở rộng backbone động cho mỗi task mới mà không làm nhiễu (noise) các lớp cũ nhờ khởi tạo trọng số chéo bằng 0.
2. **Knowledge Distillation (KD) Loss:** Kết hợp Distillation Loss trên logits để bảo tồn tri thức sâu của các lớp cũ (chống quên hiệu quả).
3. **Adaptive Class Weights & Balanced Replay:** Cân bằng trọng số hàm Loss theo đúng tỷ lệ thực tế sau khi đã lồng ghép dữ liệu exemplar.
4. **FC Warmup (Two-stage Training):** Đóng băng backbone mới trong các vòng đầu tiên để lớp phân loại (FC layer) làm quen với số lượng lớp mới, tránh hỏng đặc trưng.

---

## 🏛️ Kiến trúc Phân cấp (3-Tier)

Hệ thống được thiết kế theo mô hình 3 lớp khớp với thực tế hạ tầng mạng công nghiệp:

1.  **Cloud Server**: Tổng hợp mô hình Global, quản lý kiến trúc mạng mở rộng. (Sử dụng Weight Aligning cho phương pháp iCaRL/WA).
2.  **Edge Server**: Trung tâm huấn luyện và quản lý bộ nhớ (Exemplar Memory).
    *   **WTO (Weighted Transmission Optimization)**: Tối ưu hóa băng thông bằng cách chọn lọc các Client quan trọng nhất để truyền dữ liệu dựa trên độ hiếm của lớp và trạng thái kênh truyền.
    *   **FCIL Training**: Huấn luyện tăng cường kết hợp **Replay Buffer (Herding)** và **Knowledge Distillation (KD)**.
3.  **Client (IIoT Device)**: Thu thập lưu lượng mạng (NetFlow) và gửi dữ liệu lên Edge khi được chọn qua cơ chế WTO.

---

## 📂 Cấu trúc Dự án

*   `main.py`: Script chính để khởi chạy chu trình huấn luyện Federated Class-Incremental.
*   `plot.py`: Công cụ trực quan hóa kết quả thực nghiệm (Accuracy, F1-Score) từ các file log CSV.
*   `evaluate.py`: Đánh giá chi tiết model sau huấn luyện.
*   `data/`: Chứa bộ nạp dữ liệu (`fl_dataset_loader.py`) và logic phân chia Non-IID.
*   `models/`: Kiến trúc 1-D CNN và các biến thể mạng cho Incremental Learning (DER, iCaRL, WA).
*   `federated/`: Logic điều phối Cloud - Edge - Client.
*   `incremental/`: Triển khai các thuật toán cốt lõi (WTO, Loss functions, Memory management).
*   `logs/`: Lưu trữ kết quả thực nghiệm (CSV) và các biểu đồ đầu ra (.png).

---

## 🛠️ Hướng dẫn Sử dụng

### 1. Cài đặt Môi trường
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình
Sử dụng tham số `--dataset` để chọn bộ dữ liệu (`nf_uq_nids` hoặc `nf_ton_iot`):

```bash
# Huấn luyện trên bộ NF-UQ-NIDS-v2 (21 lớp, 5 Task) với thuật toán DER tối ưu:
python main.py --dataset nf_uq_nids --method der --wto_beta 0.5

# Huấn luyện trên bộ NF-ToN-IoT-v2 (10 lớp, 5 Task) với iCaRL:
python main.py --dataset nf_ton_iot --method icarl --wto_beta 0.5

# Chạy chế độ Debug (nhanh) để kiểm tra code:
python main.py --dataset nf_ton_iot --method der --wto_beta 0.5 --epochs_base 2 --epochs_incremental 2 --debug
```

### 3. Trực quan hóa Kết quả
Sau khi huấn luyện, các file log sẽ được lưu trong thư mục `logs/`. Sử dụng `plot.py` để vẽ biểu đồ so sánh:

```bash
python plot.py
```
Biểu đồ sẽ được lưu trực tiếp vào thư mục log của từng dataset tương ứng (ví dụ: `logs/NF-UQ-NIDS-v2_acc_plot.png`).

---

## 📊 Bộ dữ liệu Hỗ trợ
*   **NF-UQ-NIDS-v2**: 21 loại nhãn, được chia thành 5 Task (Lớp cơ sở + 4 Task tăng trưởng).
*   **NF-ToN-IoT-v2**: 10 loại nhãn, được chia thành 5 Task (Mỗi task 2 lớp).

---
*Dự án được phát triển bởi TongXuanVu.*
