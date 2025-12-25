# DIGITS CLASSIFICATION - ỨNG DỤNG MẠNG NƠ-RON TÍCH CHẬP (CNN) TRONG PHÂN LOẠI CHỮ SỐ VIẾT TAY VỚI TẬP DỮ LIỆU MNIST
Dự án này tập trung nghiên cứu và ứng dụng Mạng nơ-ron tích chập (CNN) để giải quyết bài toán nhận diện và phân loại chữ số viết tay. Mô hình được huấn luyện trên tập dữ liệu chuẩn MNIST và đạt được hiệu quả vượt trội so với các phương pháp truyền thống như MLP hay SVM

## 📖 Tổng quan
Trong kỷ nguyên số hóa, bài toán nhận dạng quang học (OCR) đóng vai trò then chốt. Dự án này xây dựng một quy trình (pipeline) hoàn chỉnh từ tiền xử lý dữ liệu, thiết kế kiến trúc mạng, huấn luyện và đánh giá mô hình.
Mục tiêu chính:
+ Xây dựng mô hình CNN tự động học các đặc trưng không gian từ ảnh đầu vào.
+ Đạt độ chính xác cao trên tập dữ liệu kiểm thử.
+ So sánh hiệu năng với các mạng nơ-ron truyền thống.

## 🛠 Công cụ và Thư viện
Dự án được phát triển trên ngôn ngữ Python, sử dụng các thư viện sau:
+ Pytorch (torch): Framework chính, xây dựng kiến trúc mạng và lan truyền ngược.
+ Torchvision: Cung cấp bộ dữ liệu MNIST và các công cụ tiền xử lý ảnh (Transform)
+ Numpy: Hỗ trợ tính toán ma trận và xử lý dữ liệu đầu vào
+ Pillow (PIL): Xử lý ảnh thực tế bên ngoài cho quá trình dự đoán (Inference)
+ Tqdm: Hiển thị thanh tiến trình huấn luyện.
Môi trường phần cứng hỗ trợ tự động chuyển đổi giữa CPU và GPU (CUDA) để tăng tốc tính toán.

## 🧠 Kiến trúc Mô hình (SimpleCNN)
Mô hình được thiết kế để khai thác đặc trưng không gian 2D của ảnh chữ số. Kiến trúc cụ thể bao gồm:
+ Lớp Tích chập 1 (Conv1): 32 kernel (3x3), hàm kích hoạt ReLU
+ Lớp Pooling 1: Max Pooling (2x2) để giảm kích thước không gian
+ Lớp Tích chập 2 (Conv2): 64 kernel (3x3), hàm kích hoạt ReLU
+ Lớp Pooling 2: Max Pooling (2x2)
+ Lớp Kết nối đầy đủ (Fully Connected): Làm phẳng (Flatten) feature maps và đưa vào mạng nơ-ron để phân loại. Đầu ra: Sử dụng hàm Softmax để xác định xác suất cho 10 lớp chữ số (0-9)

## ⚙️ Quy trình Huấn luyện
+ Dữ liệu: Tập MNIST gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra, kích thước 28x28 pixel (grayscale)
+ Tiền xử lý: Chuẩn hóa giá trị pixel về [0, 1], One-hot encoding nhãn, chia Batch size = 64.
+ Hàm mất mát (Loss Function): Categorical Cross-Entropy Loss.
+ Thuật toán tối ưu (Optimizer): Adam (cho tốc độ hội tụ nhanh hơn SGD).
+ Chu kỳ huấn luyện: 20 Epochs.

## 📊 Kết quả Thực nghiệm
Sau quá trình huấn luyện và kiểm thử, mô hình đạt được các chỉ số ấn tượng:
| Tập dữ liệu | Độ chính xác (Accuracy) |
| :--- | :---: |
| **Train** | **99%** |
| **Validation** | **98.82%** |
| **Test** | **90% - 99.88%** |
So sánh với MLP (Multi-Layer Perceptron):CNN vượt trội hơn MLP (chỉ đạt 97-98%) nhờ khả năng bảo toàn cấu trúc không gian của ảnh và khả năng bất biến với các dịch chuyển nhỏ.

## 🚀 Hướng dẫn Cài đặt & Sử dụng. 
### 1. Cài đặt môi trường: 
Đồ án khuyến khích sử dụng **Micromamba** (hoặc **Conda**) để quản lý môi trường nhằm xung tránh xung đột thư viện.
+ Thiết lập môi trường:
    micromamba create -n Main_env python = 3.10 
    micromamba activate Main_env
+ Clone dự án:
    https://github.com/Wocanh-161/Digits_classification.git
+ Cài đặt thư viện:
    pip install -r requirements.txt

### 2. Huấn luyện mô hình
Chạy script huấn luyện (Trainer) để bắt đầu train mô hình trên tập MNIST. 
    python3 trainer.py
Quá trình này sẽ tự động tải dữ liệu nếu chưa có.

### 3. Kiểm thử (Prediction)
Sử dụng script predict.py để dự đoán trên ảnh tự vẽ. Lưu ý ảnh đầu vào nên có nền đen chữ trắng hoặc được tiền xử lý đảo màu tương ứng để khớp với dữ liệu MNIST.
    3.1. Chuyển ảnh cần dự đoán vào thư mục chứa tệp predict.py (Nếu không muốn, bạn có thể không cần làm bước này)
    3.2. Trong tệp predict.py, lướt xuống hàm def main, dưới #4. GỌI HÀM DỰ ĐOÁN ẢNH CỦA EM, thay "Screenshot 2025-12-12 165610.png" thành đường dẫn tương đối của ảnh cần dự đoán (Có thể dùng ảnh bạn vừa truyền vào, hoặc ảnh đã được cung cấp sẵn). Vd: "anh2.png"
    3.3. Trong Terminal của VScode, chạy lệnh sau để bắt đầu dự đoán:
        python3 predict.py

## 🔮 Hướng phát triển
Mặc dù kết quả khả quan, dự án dự kiến sẽ cải tiến thêm các hạng mục sau:
+ Data Augmentation: Áp dụng xoay ngẫu nhiên, phóng to/thu nhỏ để mô hình nhận diện tốt hơn các chữ viết nghiêng hoặc lệch.
+ Xây dựng GUI/Web: Phát triển ứng dụng Web (Streamlit/Flask) cho phép vẽ trực tiếp lên màn hình.
+ Tinh chỉnh tham số (Hyperparameter Tuning): Thử nghiệm Learning Rate và số lượng bộ lọc khác nhau.
+ Phân tích sai số: Sử dụng Confusion Matrix để phân tích các cặp số hay bị nhầm lẫn.

## Cấu trúc thư mục:
Digits_classification/
│
├── configs
│   └── config.yaml              # File cấu hình (hyperparameters, đường dẫn, batch size, ...)
|
├── data
|   └── MNIST
|       └── raw             # Dữ liệu MNIST gốc (ảnh & nhãn)
|
├──src/
|    ├── data/
|    │   ├── dataloader.py        # Load và tiền xử lý 
|    |    └── MNIST/
|    |      └── raw/             # Dữ liệu MNIST gốc (ảnh & nhãn)
|    │
|    ├── losses/
|    |    └── loss.py              # Định nghĩa hàm mất mát
|    │
|    └── models/
|       └── model.py             # Định nghĩa kiến trúc mạng CNN
|
├── predict.py
├── README.md                    # Tài liệu mô tả dự án
├── requirements.txt             # Danh sách thư viện cần thiết
└── trainer.py                   # Script huấn luyện mô hình CNN

## 👥 Tác giả
Nhóm thực hiện:
1. Võ Ngọc Bảo
2. Nguyễn Quốc Anh
3. Đinh Ngọc Bích
4. Trương Thị Ngọc Hà
5. Nguyễn Hoàng Châu
6. Đỗ Xuân Huy
Giảng viên hướng dẫn thực hành: Lê Đức Khoan

# BẢNG PHÂN CÔNG:
<img width="1869" height="811" alt="image" src="https://github.com/user-attachments/assets/54731508-5fb4-4a31-893d-16d9e389057e" />
