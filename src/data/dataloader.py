import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Chuyển nhãn thành one-hot
def one_hot_encode(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Định nghĩa các chuyển đổi (transformations)
transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển ảnh thành Tensor
])

# Tải bộ dữ liệu MNIST (không tải lại, áp dụng transformations)
def load_data():
    # Tải bộ dữ liệu MNIST (chỉ tập train, vì testset có sẵn)
    trainset = datasets.MNIST(root='./src/data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./src/data', train=False, download=True, transform=transform)

    # Chia tập huấn luyện thành train và validation (80% train, 20% val)
    train_size = int(0.8 * len(trainset))  # Tính kích thước train
    val_size = len(trainset) - train_size  # Tính kích thước validation
    train_data, val_data = random_split(trainset, [train_size, val_size])  # Chia tập

    # Tạo DataLoader cho train, validation và test
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = DataLoader(val_data, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Chuyển nhãn thành one-hot và trả về DataLoader với nhãn đã chuyển
    def transform_labels(batch):
        images, labels = batch
        labels_one_hot = one_hot_encode(labels)  # Chuyển nhãn thành one-hot
        return images, labels_one_hot

    # Áp dụng transform_labels cho mỗi batch
    trainloader = map(transform_labels, trainloader)
    valloader = map(transform_labels, valloader)
    testloader = map(transform_labels, testloader)

    return trainloader, valloader, testloader
