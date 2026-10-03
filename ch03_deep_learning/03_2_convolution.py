"""03-2. CNN 이해하기 — Convolution은 02-3의 Sobel과 같은 계산이다

실행 (저장소 루트에서, python data/make_parts.py 후):
    python ch03_deep_learning/03_2_convolution.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
img = cv2.imread(str(ROOT / "data" / "parts" / "normal" / "000_ok.png"), cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# 1. Sobel 커널을 Convolution 층의 가중치로 넣어 본다
sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    conv.weight[0, 0] = sobel_x
    out = conv(torch.from_numpy(img).float()[None, None])[0, 0].numpy()
ref = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
print("Conv2d 가중치 shape:", tuple(conv.weight.shape), "(출력 채널, 입력 채널, 높이, 너비)")
print(f"Conv2d 결과와 cv2.Sobel 결과의 최대 차이: {np.abs(out - ref).max():.6f}")

# 2. 같은 일을 Fully Connected 층으로 하면 파라미터가 몇 개 필요할까
conv16 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
conv_params = sum(p.numel() for p in conv16.parameters())
fc_params = (h * w) * (16 * h * w) + 16 * h * w          # 입력 화소 전부 x 출력 화소 전부 + bias
print(f"\n입력 {h}x{w} 흑백 → 특징 지도 16장 ({h}x{w})")
print(f"  Convolution 3x3: 파라미터 {conv_params:,}개")
print(f"  Fully Connected : 파라미터 {fc_params:,}개 (float32로 {fc_params * 4 / 1024 ** 4:,.0f} TB)")

# 3. 블록을 지날 때마다 특징 지도는 작아지고, 한 화소가 보는 범위(Receptive Field)는 넓어진다
x = torch.zeros(1, 3, 168, 224)
rf, jump = 1, 1
print("\n블록   출력 shape          Receptive Field")
for i in range(4):
    block = nn.Sequential(nn.Conv2d(x.shape[1], 16 * 2 ** min(i, 2), 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    x = block(x)
    rf += 2 * jump              # 3x3 Convolution: 양쪽으로 한 칸씩 넓어진다
    rf += 1 * jump              # 2x2 MaxPool
    jump *= 2                   # Pooling 뒤에는 한 칸이 원본 2배 거리
    print(f"  {i + 1}    {str(tuple(x.shape)):<20}{rf}x{rf} 화소")
