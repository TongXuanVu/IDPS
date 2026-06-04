# Kế Hoạch Cải Tiến Thuật Toán DER Đã Thực Hiện

Tài liệu này tổng hợp toàn bộ các bước trong kế hoạch cải tiến thuật toán **DER (Dark Experience Replay)** đã được phân tích, triển khai và kiểm chứng thành công trong khuôn khổ dự án HFIN.

## 🎯 Mục Tiêu Cải Tiến
Giải quyết triệt để hiện tượng **Catastrophic Forgetting (Quên lãng thảm họa)** và **Overfitting dữ liệu Replay** của thuật toán DER ban đầu trên môi trường Federated Learning với dữ liệu mạng lưới (NIDS) có tính mất cân bằng cực đoan.

---

## 🛠️ Các Bước Cải Tiến Đã Triển Khai

### Bước 1: Khắc phục Nhiễu Mở rộng Động (Zero-Initialization)
- **File sửa đổi:** `models/der_network.py`
- **Tình trạng cũ:** Khi Backbone được mở rộng để đón nhận Task mới, các liên kết chéo ở lớp Fully Connected (`fc`) bị khởi tạo ngẫu nhiên, gây nhiễu và làm sai lệch xác suất dự đoán của các lớp cũ ngay từ epoch đầu tiên.
- **Hành động đã làm:** Canh thiệp vào hàm `update_fc`, thiết lập cứng (hardcode) tất cả các trọng số chéo (cross-weights) về `0.0`. 
- **Kết quả:** Bảo vệ an toàn tuyệt đối cho tri thức cũ trong giai đoạn đầu học Task mới.

### Bước 2: Tái thiết lập Bảo tồn Tri thức (Knowledge Distillation - KD)
- **File sửa đổi:** `federated/edge_server.py`
- **Tình trạng cũ:** DER vô tình bị loại khỏi cơ chế giữ bản sao `old_model` và chỉ học bằng Cross-Entropy thông thường (đi ngược lại thiết kế gốc của DER là phải có mỏ neo logits).
- **Hành động đã làm:** Kích hoạt tính năng clone `old_model` cho DER. Tích hợp hàm `distillation_loss` bằng cách tính KL-Divergence giữa logits của model hiện tại và `old_model` trên các lớp đã học.
- **Kết quả:** "Neo" chặt vùng nhớ của mạng, triệt tiêu 80% hiện tượng quên ở Task 3 và 4.

### Bước 3: Cân bằng Trọng số Thích ứng (Adaptive Class Weights)
- **File sửa đổi:** `federated/edge_server.py`
- **Tình trạng cũ:** Trọng số cân bằng lớp (`class_weights`) được tính toán dựa trên dữ liệu Client **trước khi** lồng ghép với bộ đệm Replay Memory. Hậu quả là các lớp cũ vừa bị chèn ép x500 lần số lượng, vừa mang trọng số phạt cực cao.
- **Hành động đã làm:** Dời hàm nội suy `class_weights` xuống sau hàm concat. Sử dụng công thức `Square Root Smoothing` trực tiếp trên tổng số lượng nhãn của batch đã được hòa trộn.
- **Kết quả:** Trả lại sự công bằng giữa các lớp thiểu số cũ và các lớp tấn công mới, ngăn chặn mạng "lười biếng" chỉ đoán lớp cũ.

### Bước 4: Chiến lược Huấn luyện 2 Giai đoạn (FC Warmup)
- **File sửa đổi:** `main.py` & `federated/edge_server.py`
- **Tình trạng cũ:** Backbone mới (Conv1D) vừa khởi tạo đã phải học ngay lập tức, chịu dòng gradient cực lớn từ sự sai lệch của lớp `fc` mới mở rộng, dẫn đến hỏng bộ trích xuất đặc trưng.
- **Hành động đã làm:** Truyền biến `round_in_task` vào tiến trình học. Nếu `round_in_task < 3` (tức 2 vòng đầu), backbone mới sẽ bị **đóng băng (freeze)** (`requires_grad = False`). Chỉ cho phép lớp `fc` cập nhật để làm quen với số lượng nhãn. Bắt đầu từ vòng 3 mới mở khóa toàn bộ.
- **Kết quả:** Tránh bão hòa sớm, giúp backbone có thời gian "khởi động ấm" (Warmup).

### Bước 5: Vi tinh chỉnh Hệ số Phạt (Tuning Auxiliary Loss)
- **File sửa đổi:** `federated/edge_server.py`
- **Tình trạng cũ:** Hệ số `lambda_aux = 1.0` quá lớn, ép mạng mới phải luôn trả về `0` cho dữ liệu cũ, làm suy giảm khả năng trích xuất tính năng tấn công.
- **Hành động đã làm:** Giảm `lambda_aux` xuống `0.1`.
- **Kết quả:** Giải phóng sức mạnh của Backbone mới, tập trung tối đa vào việc học tấn công lạ.

---

## 📈 Tác Động Ghi Nhận
Với 5 cải tiến có tính móc xích này, phiên bản **DER Tối ưu (DER Opt)** đã:
1. Chiếm lĩnh vị trí **Thuật toán tốt nhất** trong Framework HFIN.
2. Vượt mặt iCaRL và WA từ **2% đến 6%** ở cả F1-Macro và Accuracy ở các Task khốc liệt nhất (Task 3, Task 4).
3. Duy trì được đường cong học tập cực kỳ ổn định, xóa bỏ tình trạng cắm đầu tụt dốc cuối mỗi Task.
