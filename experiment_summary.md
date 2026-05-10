# Kịch bản Thực nghiệm Hệ thống HFIN với Dữ liệu NF-ToN-IoT-v2 và NF-UQ-NIDS-v2

Tài liệu này tóm tắt toàn bộ cấu hình, kiến trúc và kịch bản huấn luyện cho hệ thống HFIN (Hybrid Federated Incremental Learning) áp dụng trên tập dữ liệu Network Intrusion Detection System.

## 1. Thông tin Dữ liệu (Datasets)
Dự án sử dụng 2 bộ dữ liệu chuẩn NetFlow (V2) để đánh giá hệ thống:

**A. NF-ToN-IoT-v2**
*   **Đặc trưng**: 39 đặc trưng số (numeric features).
*   **Tổng số lớp học**: 10 lớp (0 - 9).
    *   **Lớp 0**: Benign (Lưu lượng sạch).
    *   **Lớp 1-9**: Các loại tấn công (DDoS, DoS, v.v.).
*   **Global Test Set**: 3.3 triệu mẫu.

**B. NF-UQ-NIDS-v2**
*   **Đặc trưng**: 39 đặc trưng số (đã đồng bộ định dạng với ToN-IoT).
*   **Tổng số lớp học**: 21 lớp (0 - 20).
    *   **Lớp 0-4**: Base classes.
    *   **Lớp 5-20**: Các loại tấn công mới xuất hiện theo từng giai đoạn.
*   **Global Test Set**: 4.8 triệu mẫu.

## 2. Kiến trúc Hệ thống (3-Tier Hierarchy)
Hệ thống được thiết lập theo cấu trúc phân cấp chuẩn của bài báo:
*   **Cloud Server**: 01 Server trung tâm (Thực hiện FedAvg để tổng hợp kiến thức toàn cầu).
*   **Edge Servers**: **03 Servers** (Mỗi server quản lý một vùng mạng, thực hiện tổng hợp cục bộ).
*   **IIoT Clients**: **60 Clients** (Chia đều 20 clients cho mỗi Edge Server).
    *   *Dữ liệu từ 3 tệp Parquet gốc của Edge được chia nhỏ thành 20 phần cho 20 clients tương ứng.*

## 3. Kịch bản Incremental Learning (5 Tasks)
Quá trình huấn luyện chia thành 5 giai đoạn liên tiếp. Mỗi giai đoạn, mô hình sẽ tiếp xúc với các loại tấn công mới.

### Kịch bản NF-ToN-IoT-v2 (2-2-2-2-2)
| Giai đoạn | Lớp Học Mới | Mô tả |
| :--- | :--- | :--- |
| **Task 0 (Base)** | [0, 1] | Học lớp Benign và tấn công đầu tiên. |
| **Task 1** | [2, 3] | Thêm 2 loại tấn công mới. |
| **Task 2** | [4, 5] | Thêm 2 loại tấn công mới. |
| **Task 3** | [6, 7] | Thêm 2 loại tấn công mới. |
| **Task 4** | [8, 9] | Thêm 2 loại tấn công cuối cùng. |

### Kịch bản NF-UQ-NIDS-v2 (5-4-4-4-4)
| Giai đoạn | Lớp Học Mới | Mô tả |
| :--- | :--- | :--- |
| **Task 0 (Base)** | [0, 1, 2, 3, 4] | Học 5 lớp cơ sở. |
| **Task 1** | [5, 6, 7, 8] | Thêm 4 loại tấn công mới. |
| **Task 2** | [9, 10, 11, 12] | Thêm 4 loại tấn công mới. |
| **Task 3** | [13, 14, 15, 16] | Thêm 4 loại tấn công mới. |
| **Task 4** | [17, 18, 19, 20]| Thêm 4 loại tấn công cuối cùng. |

## 4. Các Phương pháp Đối chiếu (Benchmarking)
Thí nghiệm so sánh 3 thuật toán cốt lõi với 2 biến thể WTO:

1.  **iCaRL**: Sử dụng Exemplar Replay và Knowledge Distillation.
2.  **WA (Weight Aligning)**: iCaRL kết hợp với kỹ thuật hiệu chỉnh trọng số lớp cuối (Khuyên dùng).
3.  **DER (Dark Experience Replay)**: Sử dụng logit-level replay để giữ kiến thức cũ.

**Biến thể WTO (Weighted Transmission Optimization):**
*   **With WTO**: Kích hoạt bộ lọc Clients dựa trên tầm quan trọng của dữ liệu (`--wto_beta 0.5`).
*   **No WTO**: Không lọc, truyền tải toàn bộ cập nhật (`--wto_beta 0`).

## 5. Tham số Huấn luyện Mặc định (Hyperparameters)
*   **Batch Size**: 512 (Tối ưu cho Kaggle GPU).
*   **Learning Rate**: 0.01 (Base Task) / 0.02 (Incremental Tasks).
*   **Local Epochs**: 1.
*   **Optimizer**: SGD (Momentum=0.9, Weight Decay=5e-4).
*   **Memory Size**: 2000 samples (Tổng cộng toàn hệ thống).
*   **WTO Beta**: 0.5.

## 6. Chỉ số Đánh giá (Metrics)
*   **Accuracy / F1-Score**: Độ chính xác trên tập Test toàn cục sau mỗi Task.
*   **Forgetting Measure**: Mức độ sụt giảm hiệu suất trên các lớp cũ.
*   **Communication Overhead**: Hiệu quả tiết kiệm băng thông của cơ chế WTO.

---
*Tài liệu được tạo tự động bởi Antigravity AI Assistant cho dự án HFIN-IDPS.*
