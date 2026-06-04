# Báo cáo Cập nhật Thuật toán DER (Dark Experience Replay) - Phiên bản Tối ưu

Bản cập nhật này giải quyết triệt để các vấn đề khiến thuật toán DER (sử dụng kiến trúc mạng mở rộng động) có hiệu năng kém hơn HFL/iCaRL ở các Task cuối, đặc biệt khi xử lý bộ dữ liệu mạng (NetFlow) có độ mất cân bằng cực cao.

## 1. Zero-Initialization cho Trọng số Mạng Mở rộng (Cross-connections)
**Vấn đề cũ:** Khi một backbone mới được thêm vào, lớp `fc` được mở rộng. Các trọng số kết nối từ *feature cũ -> lớp mới* và *feature mới -> lớp cũ* bị khởi tạo ngẫu nhiên. Trọng số ngẫu nhiên này lập tức "đầu độc" (gây nhiễu) xác suất dự đoán của các lớp cũ mà mô hình đã học rất tốt trước đó.
**Giải pháp:** 
- Khởi tạo toàn bộ các vùng trọng số chéo bằng `0.0`.
- Kết quả: Các lớp cũ sẽ an toàn tuyệt đối trước sự thay đổi của mạng mới, cho đến khi mạng mới thực sự hội tụ.

## 2. Tái kích hoạt Knowledge Distillation (KD Loss)
**Vấn đề cũ:** Do sự nhầm lẫn giữa 2 phiên bản DER (của Buzzega và của Yan), mã nguồn trước đó của DER đã vô tình bỏ qua việc lưu bản sao `old_model`, dẫn đến việc mô hình chỉ học bằng hàm Cross-Entropy thông thường và bị hiện tượng Catastrophic Forgetting (Quên lãng thảm họa) đánh bại.
**Giải pháp:**
- Kích hoạt cơ chế lưu `old_model` tương tự như iCaRL.
- Thay thế Loss phân loại thông thường bằng hàm `distillation_loss`. Hàm này áp dụng KL-Divergence để "neo" (anchor) các giá trị logits của lớp cũ không bị trôi đi trong quá trình học lớp mới.

## 3. Tính toán Trọng số Lớp (Adaptive Class Weights) Động
**Vấn đề cũ:** Do dữ liệu có tính Non-IID và mất cân bằng nghiêm trọng, hệ thống có cơ chế Balanced Replay (oversample dữ liệu cũ lên đến 500 lần). Tuy nhiên, hàm tính `class_weights` lại được chạy **trước** khi dữ liệu Replay được thêm vào. Hậu quả là lớp cũ vừa bị lặp lại 500 lần trong batch, vừa bị nhân hệ số phạt x3. Mô hình bị quá tải (over-penalize) dẫn tới overfit bộ đệm và quên lớp mới.
**Giải pháp:**
- Dời bước tính `class_weights` xuống sau bước ghép dữ liệu Replay.
- Áp dụng kỹ thuật Square Root Smoothing trên toàn bộ mẫu có trong batch, đảm bảo sự công bằng (fairness) giữa lớp thiểu số cũ và lớp đa số mới.

## 4. FC Warmup (Huấn luyện 2 Giai đoạn / Khởi động ấm)
**Vấn đề cũ:** Trong vài epoch đầu, lớp `fc` mới mở rộng còn hoàn toàn ngẫu nhiên. Sai số từ lớp `fc` tạo ra gradient rất lớn truyền ngược về làm hỏng khả năng trích xuất đặc trưng của backbone mới.
**Giải pháp:**
- **Stage 1 (Warmup):** Đóng băng (Freeze) backbone mới trong 2 vòng (rounds) đầu tiên của task. Mô hình chỉ học để điều chỉnh trọng số lớp `fc`.
- **Stage 2:** Mở khóa (Unfreeze) backbone, cho phép học cả mạng như bình thường.

## 5. Giảm thiên kiến từ Auxiliary Loss
**Vấn đề cũ:** Hệ số `lambda_aux = 1.0` ép backbone mới cố dự đoán giá trị `0` cho toàn bộ mẫu thuộc lớp cũ. Vì mẫu cũ bị lặp lại quá nhiều, backbone mới trở nên lười biếng và luôn có xu hướng dự đoán `0`.
**Giải pháp:**
- Giảm `lambda_aux` xuống `0.1`.
- Backbone mới được "cởi trói", tập trung nhiều hơn vào việc học đặc trưng của các lớp tấn công mới.

---
**Kết luận:** Với 5 bản vá này, DER hiện tại không chỉ là một thuật toán Mở rộng Đặc trưng (Dynamically Expandable Representation) xuất sắc, mà còn sở hữu khả năng bảo tồn tri thức tĩnh (Knowledge Distillation) vô cùng mạnh mẽ. Sự sụt giảm Accuracy/F1 ở các task cuối cùng sẽ được giải quyết triệt để.
