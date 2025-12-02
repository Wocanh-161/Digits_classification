# src/data/dataloader.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64):
    """
    Hàm này thực hiện toàn bộ việc tải, chia và tạo DataLoaders cho bộ MNIST.
    """
    
    # 1. Định nghĩa các phép biến đổi (Transforms)
    # Dữ liệu MNIST là ảnh (PIL Image), chúng ta cần:
    # - ToTensor(): Chuyển ảnh thành dạng Tensor (một mảng số) mà PyTorch hiểu được.
    #               Đồng thời, nó tự động chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0].
    # - Normalize((0.5,), (0.5,)): Chuẩn hóa thêm (optional nhưng nên làm).
    #               Nó đưa pixel về khoảng [-1.0, 1.0] bằng cách (giá_trị - 0.5) / 0.5.
    #               Giúp mô hình hội tụ nhanh hơn. (0.5,) là tuple chứa giá trị mean và std.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. Tải bộ dữ liệu Train và Test
    # Tải bộ train (60,000 ảnh)
    # - root='./data': Thư mục để lưu dữ liệu MNIST.
    # - train=True: Chỉ định tải bộ dữ liệu huấn luyện.
    # - download=True: Tự động tải về nếu chưa có trong thư mục './data'.
    # - transform=transform: Áp dụng các phép biến đổi đã định nghĩa ở trên.
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Tải bộ test (10,000 ảnh)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. Chia bộ Train thành Train và Validation (Tỉ lệ 6:2:2)
    # Tổng cộng ta có 60,000 train + 10,000 test = 70,000 ảnh.
    # Tỉ lệ 6:2:2 (Train:Val:Test) tương đương 42,000 : 14,000 : 14,000.
    # Tuy nhiên, bộ test có sẵn 10,000 ảnh (tỉ lệ ~14.3%).
    # Chúng ta sẽ giữ nguyên bộ test (10k) và chia bộ train (60k) theo tỉ lệ 8:2 (48k train, 12k val).
    # Như vậy, tỉ lệ cuối cùng sẽ là 48:12:10 (~ 6.8 : 1.7 : 1.4), khá gần với yêu cầu 6:2:2.
    
    train_size = int(0.8 * len(full_train_dataset))  # 80% của 60,000 = 48,000
    val_size = len(full_train_dataset) - train_size      # 20% của 60,000 = 12,000

    # Dùng hàm random_split để chia
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 4. Tạo các đối tượng DataLoader
    # DataLoader sẽ lấy dữ liệu từ các 'dataset' và đóng gói chúng thành các 'batch'.
    # - batch_size: Số lượng mẫu dữ liệu trong mỗi batch.
    # - shuffle=True (cho train_loader): Xáo trộn dữ liệu *chỉ* ở bộ huấn luyện sau mỗi epoch.
    #   Điều này cực kỳ quan trọng để mô hình học tốt và tránh bị "ghi nhớ" thứ tự dữ liệu.
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Dùng 2 tiến trình con để tải data, giúp tăng tốc
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, # Không cần xáo trộn khi đánh giá
        num_workers=2
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # Không cần xáo trộn khi đánh giá
        num_workers=2
    )
    
    print("--- Đã tải và tạo Dataloader thành công ---")
    print(f"Số lượng ảnh Train: {len(train_dataset)}")
    print(f"Số lượng ảnh Validation: {len(val_dataset)}")
    print(f"Số lượng ảnh Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# ----- Đoạn code để kiểm tra file này chạy độc lập (optional) -----
if __name__ == '__main__':
    # Đoạn này chỉ chạy khi em thực thi trực tiếp file 'dataloader.py'
    # Mục đích là để kiểm tra xem code có lỗi không
    
    # Thử tạo loader
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Thử lấy một batch dữ liệu từ train_loader
    print("\nĐang thử lấy 1 batch từ train_loader...")
    try:
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        print(f"Kiểu dữ liệu của batch ảnh: {type(images)}")
        print(f"Kích thước của batch ảnh: {images.shape}")
        print(f"Kích thước của batch nhãn: {labels.shape}")
        print(f"Một ví dụ nhãn: {labels[0]}")
        print("\n=> Test Dataloader thành công!")
        
    except Exception as e:
        print(f"\nCó lỗi xảy ra khi test Dataloader: {e}")
