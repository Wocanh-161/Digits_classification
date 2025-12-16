# predict.py
# File này dùng để tải model đã huấn luyện và dự đoán ảnh THỰC TẾ.

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image # Thư viện dùng để đọc ảnh

# 1. Import các thành phần
from src.models.model import SimpleCNN

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth" # Đường dẫn đến model em đã lưu

def predict_my_image(image_path, model):
    """
    Hàm dự đoán một ảnh đơn do người dùng cung cấp.
    """
    print(f"--- Đang dự đoán ảnh: {image_path} ---")
    
    # 1. Tải ảnh và xử lý ảnh
    try:
        # Mở ảnh bằng thư viện PIL (Pillow)
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'.")
        print("Em đã vẽ và lưu ảnh vào cùng thư mục chưa?")
        return

    # 2. Định nghĩa các phép biến đổi
    # Đây là bước quan trọng: ta phải xử lý ảnh em vẽ
    # y hệt như cách ta đã xử lý dữ liệu MNIST lúc train.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Chuyển ảnh thành ảnh xám (1 kênh)
        transforms.Resize((28, 28)),                 # Resize ảnh về 28x28 pixel
        transforms.ToTensor(),                       # Chuyển thành Tensor (giá trị 0.0 -> 1.0)
        transforms.Normalize((0.5,), (0.5,))         # Chuẩn hóa về (-1.0 -> 1.0)
    ])

    # 3. Áp dụng biến đổi
    # Biến đổi xong, ảnh sẽ có kích thước [1, 28, 28]
    image_tensor = transform(img)
    
    # 4. Model yêu cầu ảnh phải có 4 chiều (batch_size, channels, height, width)
    # Ta phải thêm 1 chiều "batch_size" ở đầu.
    # Kích thước từ [1, 28, 28] -> [1, 1, 28, 28]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # 5. Dự đoán
    with torch.no_grad():
        output = model(image_tensor) # output là vector 10 số
        
        # Dùng softmax để chuyển 10 số thành % xác suất
        probabilities = F.softmax(output, dim=1) 
        
        # Lấy ra dự đoán (là số có xác suất cao nhất)
        _, predicted_class = torch.max(probabilities, 1)
        
        # Lấy xác suất của dự đoán đó
        confidence = probabilities.max() * 100

        print(f"\n========== KẾT QUẢ ==========")
        print(f"    MODEL DỰ ĐOÁN LÀ SỐ: {predicted_class.item()}")
        print(f"    Với độ tự tin: {confidence:.2f}%")
        print("===============================")

def main():
    print(f"--- Đang tải model từ {MODEL_PATH} ---")

    # 1. Khởi tạo model (kiến trúc rỗng)
    model = SimpleCNN().to(DEVICE)
    
    # 2. Nạp trọng số (weights) đã huấn luyện vào
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Lỗi khi nạp model: {e}")
        return

    # 3. Đặt model ở chế độ đánh giá (eval)
    model.eval()
    
    # 4. GỌI HÀM DỰ ĐOÁN ẢNH CỦA EM
    predict_my_image("Screenshot 2025-12-12 165610.png", model)

if __name__ == '__main__':
    main()