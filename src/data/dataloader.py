from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Định nghĩa các chuyển đổi (transformations)
transform = transforms.Compose([
    transforms.ToTensor()  # Chuyển đổi ảnh thành Tensor
])

# Tải bộ dữ liệu MNIST (không tải lại, áp dụng transformations)
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

# Lấy một batch dữ liệu từ trainloader
images, labels = next(iter(trainloader))

# In kích thước batch
print(images.shape)

