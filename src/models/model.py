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
