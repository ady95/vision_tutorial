"""03-3. PyTorch로 이미지 분류하기 — 작은 CNN을 처음부터 학습해 정상/불량을 가른다

실행 (저장소 루트에서):
    python data/make_parts.py                                         # 시험용 (02장과 같은 180장)
    python data/make_parts.py --n 300 --seed 2027 --out data/parts_train  # 학습용 900장
    python ch03_deep_learning/03_3_classification.py
"""
import time

import torch
from torch import nn
from torchvision import transforms

from parts_cls import (LIGHTS, MEAN, STD, TRAIN_DIR, make_loader, read_rows, seed_everything, split_rows,
                       test_report, train)

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize(MEAN, STD)
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")


class SmallCNN(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool 블록 4개 + 분류기"""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))  # 168x224 → 10x14
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 10 * 14, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN()
print("파라미터 수:", sum(p.numel() for p in model.parameters()))
x = torch.zeros(1, 3, 168, 224)
for i, layer in enumerate(model.features):
    x = layer(x)
    print(f"  block {i + 1} 출력: {tuple(x.shape)}")

# 1. 데이터 — 학습용 900장을 학습 720 / 검증 180으로 나눈다
rows = read_rows(TRAIN_DIR)
train_rows, val_rows = split_rows(rows)
print(f"\n학습 {len(train_rows)}장, 검증 {len(val_rows)}장 (조명 {len(LIGHTS)}종, 정상:불량 = 1:1)")

# 2. 실험 A — normal 조명으로만 학습
seed_everything(0)
normal_train = [r for r in train_rows if r["light"] == "normal"]
normal_val = [r for r in val_rows if r["light"] == "normal"]
model = SmallCNN()
acc, sec = train(model, make_loader(TRAIN_DIR, normal_train, norm, shuffle=True),
                 make_loader(TRAIN_DIR, normal_val, norm), epochs=15, lr=1e-3, device=device, log=False)
print(f"\n[실험 A] normal 조명 {len(normal_train)}장으로 학습: 검증 정확도 {acc:.3f}, 학습 {sec:.1f}초")
test_report(model, device, norm)

# 3. 실험 B — 세 조명 모두로 학습
seed_everything(0)
model = SmallCNN()
print(f"\n[실험 B] 세 조명 {len(train_rows)}장으로 학습")
acc, sec = train(model, make_loader(TRAIN_DIR, train_rows, norm, shuffle=True),
                 make_loader(TRAIN_DIR, val_rows, norm), epochs=15, lr=1e-3, device=device)
print(f"  → 검증 정확도 {acc:.3f}, 학습 {sec:.1f}초")
test_report(model, device, norm)

# 4. GPU와 CPU — 같은 학습 1에폭에 걸리는 시간
loader = make_loader(TRAIN_DIR, train_rows, norm, shuffle=True)
for dev in (["cuda", "cpu"] if device == "cuda" else ["cpu"]):
    m = SmallCNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t = time.perf_counter()
    for xb, yb in loader:
        opt.zero_grad()
        nn.functional.cross_entropy(m(xb.to(dev)), yb.to(dev)).backward()
        opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"\n1에폭({len(train_rows)}장) 학습 시간 {dev}: {time.perf_counter() - t:.1f}초", end="")
print()
