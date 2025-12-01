# loss.py
import torch.nn as nn

def get_loss_function():
    """
    Trả về hàm loss phù hợp cho bài toán phân loại chữ số.
    Vì model trả về logits (chưa qua Softmax), ta dùng CrossEntropyLoss.
    """
    return nn.CrossEntropyLoss()
