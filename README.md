# HFIN: Hierarchical Federated Class-Incremental Learning for NIDS

HFIN (Hierarchical Federated Class-Incremental Learning) là một framework học máy phân tán được thiết kế chuyên biệt cho Hệ thống Phát hiện Xâm nhập Mạng (NIDS) trong môi trường Industrial IoT (IIoT). 

Framework này giải quyết đồng thời hai thách thức lớn: **Bảo mật dữ liệu** (thông qua Federated Learning) và **Thích ứng liên tục** (thông qua Class-Incremental Learning) mà không gây ra hiện tượng quên lãng thảm họa.

---

## 🏛️ Kiến trúc Phân cấp (3-Tier)

Hệ thống được thiết kế theo mô hình 3 lớp khớp với thực tế hạ tầng mạng công nghiệp:

1.  **Cloud Server**: Tổng hợp mô hình Global, thực hiện **Weight Aligning (WA)** để cân bằng trọng số giữa các lớp cũ và mới, giảm thiểu thiên kiến (bias) sau mỗi giai đoạn tăng trưởng.
2.  **Edge Server**: Trung tâm huấn luyện và quản lý bộ nhớ.
    *   **WTO (Weighted Transmission Optimization)**: Tối ưu hóa băng thông bằng cách chọn lọc các Client quan trọng nhất để truyền dữ liệu dựa trên độ hiếm của lớp và trạng thái kênh truyền.
    *   **FCIL Training**: Huấn luyện tăng cường kết hợp **Replay Buffer (Herding)** và **Knowledge Distillation (KD)** để duy trì tri thức cũ.
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
# Huấn luyện trên bộ NF-UQ-NIDS-v2 (21 lớp, 5 Task)
python main.py --dataset nf_uq_nids --method der --wto_type WTO

# Huấn luyện trên bộ NF-ToN-IoT-v2 (10 lớp, 5 Task)
python main.py --dataset nf_ton_iot --method der --wto_type WTO
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
