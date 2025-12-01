import sys
sys.path.append('src')  # Thêm thư mục src vào sys.path

# trainer.py
from models.model import CNN  # Import CNN model
from dataloader import load_data  # Import data loader
from loss import get_loss_function

import torch
import torch.optim as optim
import torch.nn as nn

# Hàm huấn luyện mô hình
def train_model(model, trainloader, criterion, optimizer, device, num_epochs=5):
    pass
# Hàm đánh giá mô hình
def evaluate_model(model, testloader, device):
    pass

# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, testloader = load_data()
    criterion = get_loss_function()

