"""03-1. 왜 Deep Learning이 필요한가 — 뉴런 하나와 작은 신경망

실행:
    python ch03_deep_learning/03_1_neuron.py
"""
import torch
from torch import nn

# 뉴런 하나: 입력 3개에 가중치를 곱해 더하고(+ bias), 활성화 함수를 통과시킨다
neuron = nn.Linear(in_features=3, out_features=1)
with torch.no_grad():
    neuron.weight[:] = torch.tensor([[0.5, -1.0, 2.0]])
    neuron.bias[:] = 0.1
x = torch.tensor([[1.0, 2.0, 3.0]])
z = neuron(x)
print(f"가중합 z = {z.item():.2f} | ReLU(z) = {torch.relu(z).item():.2f} | Sigmoid(z) = {torch.sigmoid(z).item():.4f}")
print(f"z가 -4.6이었다면 → ReLU {torch.relu(-z).item():.2f} | Sigmoid {torch.sigmoid(-z).item():.4f}")

# 층을 쌓은 신경망: 입력 7개(02-6의 특징값) → 은닉층 16 → 출력 2(정상/불량)
mlp = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 2))
print(mlp)
print("파라미터 수:", sum(p.numel() for p in mlp.parameters()))
