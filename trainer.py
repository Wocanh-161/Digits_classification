# trainer.py
# File chính, kết hợp Dataloader và Model để huấn luyện.

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Thư viện để tạo thanh tiến trình (Progress Bar) [cite: 386]

# 1. Import các thành phần từ các file .py khác
# Import hàm lấy data loader từ file dataloader.py
from src.data.dataloader import get_data_loaders
# Import class model từ file model.py
from src.models.model import SimpleCNN

# --- Cấu hình huấn luyện (Giống 'Load configs' trong slide) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3  # Tốc độ học (lr=1e-3) [cite: 207]
BATCH_SIZE = 64
NUM_EPOCHS = 10       # Chúng ta sẽ huấn luyện trong 10 vòng (epoch)

print(f"--- Đang sử dụng thiết bị: {DEVICE} ---")

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Hàm huấn luyện model qua 1 EPOCH (1 vòng lặp qua toàn bộ data).
    Đây chính là "Loop through data" trong slide của em.
    """
    model.train()  # Đặt model ở chế độ train() [cite: 224, 252]
    running_loss = 0.0
    
    # Dùng tqdm để tạo thanh tiến trình (progress bar) [cite: 383]
    loop = tqdm(loader, desc=f"Training Epoch")
    
    for batch_idx, (inputs, labels) in enumerate(loop):
        # Chuyển dữ liệu lên device (CPU hoặc GPU) [cite: 226, 256]
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # --- 5 bước huấn luyện quan trọng (xem slide 'Luồng huấn luyện đầy đủ') ---
        # 1. Zero gradients [cite: 227, 257, 483]
        optimizer.zero_grad()
        
        # 2. Forward pass: Đưa data qua model [cite: 228, 258, 484]
        outputs = model(inputs)
        
        # 3. Compute loss: Tính toán lỗi [cite: 229, 260, 486]
        loss = criterion(outputs, labels)
        
        # 4. Backpropagation: Lan truyền ngược lỗi [cite: 230, 261, 488]
        loss.backward()
        
        # 5. Optimizer step: Cập nhật trọng số [cite: 230, 263, 488]
        optimizer.step()
        
        # Cập nhật thông tin loss lên thanh tqdm
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    return avg_loss

def validate_one_epoch(model, loader, criterion):
    """
    Hàm đánh giá model trên bộ validation.
    """
    model.eval()  # Đặt model ở chế độ eval() (quan trọng!)
    val_loss = 0.0
    correct = 0
    total = 0
    
    # torch.no_grad() tắt việc tính toán gradient, giúp tiết kiệm bộ nhớ và tăng tốc
    with torch.no_grad():
        loop = tqdm(loader, desc="Validating")
        for inputs, labels in loop:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Tính độ chính xác (Accuracy)
            # outputs.max(1) sẽ trả về (giá_trị_lớn_nhất, chỉ_số_lớn_nhất)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
    avg_loss = val_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    """
    Hàm main, điều phối toàn bộ quá trình.
    """
    # 1. Load Dataset & Dataloader [cite: 189]
    train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # 2. Initialize model, loss, optimizer [cite: 190, 202]
    model = SimpleCNN().to(DEVICE)
    
    # Dùng CrossEntropyLoss cho bài toán phân loại [cite: 213]
    criterion = nn.CrossEntropyLoss()
    
    # Dùng Adam optimizer [cite: 215]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 3. Training Loop (Vòng lặp huấn luyện chính) [cite: 192, 221] ---
    print("\n--- Bắt đầu huấn luyện ---")
    
    best_val_accuracy = 0.0 # Lưu lại độ chính xác tốt nhất
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        
        # Huấn luyện
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        
        # Đánh giá
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch + 1} kết thúc:")
        print(f"\t- Train Loss: {train_loss:.4f}")
        print(f"\t- Val Loss: {val_loss:.4f}")
        print(f"\t- Val Accuracy: {val_accuracy:.2f}%")
        
        # 4. Save model checkpoints [cite: 194, 273]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"\t>>> Đạt độ chính xác tốt nhất mới. Đang lưu model...")
            torch.save(model.state_dict(), "best_model.pth")
            
    print("\n--- Huấn luyện hoàn tất ---")
    print(f"Độ chính xác tốt nhất trên bộ Validation: {best_val_accuracy:.2f}%")

    # 5. Đánh giá cuối cùng trên bộ TEST (Optional nhưng nên làm)
    print("\n--- Đang đánh giá trên bộ Test ---")
    # Tải lại model tốt nhất đã lưu
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_accuracy = validate_one_epoch(model, test_loader, criterion)
    print(f"Kết quả trên bộ Test:")
    print(f"\t- Test Loss: {test_loss:.4f}")
    print(f"\t- Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()
