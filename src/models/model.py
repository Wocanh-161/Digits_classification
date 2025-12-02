import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa class CNN (để trainer.py có thể import được)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Hiện tại chưa cần code kiến trúc, chỉ cần 'pass' để định nghĩa hàm hợp lệ
        pass 

    def forward(self, x):
        # Hàm forward cũng có thể để trống
        return x

# src/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Một kiến trúc CNN đơn giản cho bài toán MNIST.
    Luồng dữ liệu sẽ là: (Conv -> ReLU -> MaxPool) -> (Conv -> ReLU -> MaxPool) -> (Linear -> ReLU) -> (Linear)
    """
    
    def __init__(self):
        """
        Hàm này dùng để khai báo các layers.
        """
        super(SimpleCNN, self).__init__()
        
        # --- Lớp Tích chập (Convolutional) ---
        # Ảnh MNIST là 1 kênh (ảnh xám).
        # conv1: Nhận vào 1 kênh, trả ra 10 'feature map' (bộ lọc), kích thước kernel là 5x5.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # pool1: Lớp gộp, lấy giá trị lớn nhất trong cửa sổ 2x2.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv2: Nhận vào 10 kênh (từ conv1), trả ra 20 feature map, kernel 5x5.
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # pool2: Tương tự pool1.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Lớp Kết nối đầy đủ (Fully Connected / Linear) ---
        # Kích thước ảnh ban đầu là 28x28.
        # Qua conv1 (kernel 5) -> 24x24
        # Qua pool1 (kernel 2) -> 12x12
        # Qua conv2 (kernel 5) -> 8x8
        # Qua pool2 (kernel 2) -> 4x4
        # => Kích thước cuối cùng là 20 kênh (từ conv2) * 4 * 4 = 320
        #
        # fc1: Nhận vào 320 features, trả ra 50 features.
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        
        # fc2: Lớp output. Nhận vào 50 features, trả ra 10 features.
        # (Vì chúng ta có 10 lớp: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """
        Hàm này định nghĩa luồng dữ liệu đi qua các layers.
        x có kích thước: [batch_size, 1, 28, 28]
        """
        
        # Tầng 1: Conv + ReLU + Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # Kích thước lúc này: [batch_size, 10, 12, 12]
        
        # Tầng 2: Conv + ReLU + Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # Kích thước lúc này: [batch_size, 20, 4, 4]
        
        # "Làm phẳng" (Flatten) dữ liệu để chuẩn bị cho lớp Linear
        # Chuyển từ [batch_size, 20, 4, 4] thành [batch_size, 320]
        x = x.view(-1, 320) # -1 để PyTorch tự tính batch_size
        
        # Tầng 3: Linear + ReLU
        x = self.fc1(x)
        x = F.relu(x)
        # Kích thước lúc này: [batch_size, 50]
        
        # Tầng 4: Lớp output (Linear)
        # Chúng ta không dùng ReLU ở đây, vì hàm Loss (CrossEntropyLoss) sẽ tự xử lý.
        x = self.fc2(x)
        # Kích thước cuối cùng: [batch_size, 10]
        
        return x

# ----- Đoạn code để kiểm tra file này chạy độc lập (optional) -----
if __name__ == '__main__':
    # Đoạn này chỉ chạy khi em thực thi trực tiếp file 'model.py'
    
    # 1. Tạo một model
    model = SimpleCNN()
    print("--- Đã khởi tạo model SimpleCNN thành công ---")
    print("Kiến trúc model:")
    print(model)
    
    # 2. Tạo một "batch" dữ liệu giả (dummy data)
    # Giả lập 1 batch_size = 64, 1 kênh, ảnh 28x28
    dummy_input = torch.randn(64, 1, 28, 28) 
    
    # 3. Thử cho dữ liệu chạy qua model (forward pass)
    try:
        output = model(dummy_input)
        print(f"\nKích thước input giả lập: {dummy_input.shape}")
        print(f"Kích thước output của model: {output.shape}")
        
        if output.shape == (64, 10):
            print("\n=> Test Model thành công! Kích thước input/output chính xác.")
        else:
            print(f"\n=> Lỗi Test: Output shape phải là (64, 10), nhưng đang là {output.shape}")
            
    except Exception as e:
        print(f"\nCó lỗi xảy ra khi test model: {e}")
