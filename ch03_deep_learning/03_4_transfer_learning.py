"""03-4. Transfer Learning과 Data Augmentation

1) 학습 데이터 양(30·90·720장)에 따라 처음부터 학습한 모델과 사전학습 모델을 비교한다
2) Data Augmentation이 도움이 될 때와 해가 될 때를 확인한다

실행 (저장소 루트에서, 03-3의 데이터 준비 후):
    python ch03_deep_learning/03_4_transfer_learning.py
"""
import math

import torch
from torch import nn
from torchvision import models, transforms

from parts_cls import (MEAN, STD, TEST_DIR, TRAIN_DIR, labels_of, make_loader, predict, read_rows, seed_everything,
                       split_rows, test_report, train)

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize(MEAN, STD)
rows = read_rows(TRAIN_DIR)
train_rows, val_rows = split_rows(rows)
val_loader = make_loader(TRAIN_DIR, val_rows, norm)
test_rows = read_rows(TEST_DIR)
test_loader = make_loader(TEST_DIR, test_rows, norm)


def resnet18(mode):
    """mode: scratch(무작위 초기화) / frozen(사전학습 특징 고정, 마지막 층만 학습) / finetune(사전학습 전체 미세조정)"""
    weights = None if mode == "scratch" else models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    if mode == "frozen":
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)              # ImageNet 1000종 분류기 → 정상/불량 2종으로 교체
    return m


def test_acc(model):
    pred, _ = predict(model, test_loader, device)
    return (pred == labels_of(test_loader)).mean()


# 1. 데이터 양과 사전학습
print("ResNet18 파라미터 수:", sum(p.numel() for p in resnet18("scratch").parameters()))
print(f"\n{'학습 장수':>8} | {'처음부터':>8} | {'특징 고정':>8} | {'미세조정':>8}   (시험 180장 정확도)")
for n in [30, 90, 720]:
    subset = train_rows[:: len(train_rows) // n][:n]
    scores = []
    for mode, lr in [("scratch", 1e-3), ("frozen", 1e-3), ("finetune", 1e-4)]:
        seed_everything(0)
        model = resnet18(mode)
        train(model, make_loader(TRAIN_DIR, subset, norm, shuffle=True), val_loader,
              epochs=15, lr=lr, device=device, log=False)
        scores.append(test_acc(model))
    n_ok = sum(r["label"] == "ok" for r in subset)
    print(f"{n:>8} | {scores[0]:8.1%} | {scores[1]:8.1%} | {scores[2]:8.1%}   (정상 {n_ok}, 불량 {n - n_ok})")

# 2. Augmentation — normal 조명으로만 학습할 때 밝기 변형이 다른 조명을 대신할 수 있을까?
photometric = transforms.Compose([transforms.ColorJitter(brightness=0.6, contrast=0.4), norm])
normal_rows = [r for r in train_rows if r["light"] == "normal"]
normal_val = make_loader(TRAIN_DIR, [r for r in val_rows if r["light"] == "normal"], norm)
for name, tf in [("Augmentation 없음", norm), ("밝기·대비 변형", photometric)]:
    seed_everything(0)
    model = resnet18("finetune")
    train(model, make_loader(TRAIN_DIR, normal_rows, tf, shuffle=True), normal_val,
          epochs=10, lr=1e-4, device=device, log=False)
    test_report(model, device, norm, title=f"\n[normal 조명 {len(normal_rows)}장 학습 + {name}]")

# 3. Augmentation — 라벨을 지키는 변형(뒤집기)과 라벨을 망가뜨리는 변형(회전·이동)
affine = transforms.RandomAffine(degrees=25, translate=(0.15, 0.15))
torch.manual_seed(0)
broken = total = 0
for r in train_rows:
    if r["label"] != "ok":
        continue
    for _ in range(10):        # 정상품 하나에 회전·이동을 10번 걸어 보고, 02-6 규격을 벗어나는지 센다
        deg, (tx, ty), _, _ = transforms.RandomAffine.get_params(affine.degrees, affine.translate, None, None, [640, 480])
        a, dx, dy = math.radians(deg), float(r["cx"]) - 320, float(r["cy"]) - 240
        nx, ny = dx * math.cos(a) + dy * math.sin(a) + tx, -dx * math.sin(a) + dy * math.cos(a) + ty
        broken += abs(float(r["angle"]) - deg) > 8 or math.hypot(nx, ny) > 40
        total += 1
print(f"\n회전·이동 변형을 건 정상품 중 규격(각도 8도, 위치 40px)을 벗어나는 비율: {broken / total:.1%}")

flip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), norm])
geometric = transforms.Compose([affine, norm])
print(f"\n[세 조명 {len(train_rows)}장 학습] 변형별 시험 정확도 (normal / dim / gradient)")
for name, tf, seeds in [("없음", norm, [0]), ("뒤집기", flip, [0, 1]), ("회전·이동", geometric, [0, 1])]:
    for seed in seeds:
        seed_everything(seed)
        model = resnet18("finetune")
        val_acc, _ = train(model, make_loader(TRAIN_DIR, train_rows, tf, shuffle=True), val_loader,
                           epochs=10, lr=1e-4, device=device, log=False)
        accs, missed, rejected = [], 0, 0
        for light in ["normal", "dim", "gradient"]:
            lrows = read_rows(TEST_DIR, [light])
            loader = make_loader(TEST_DIR, lrows, norm)
            pred, y = predict(model, loader, device)[0], labels_of(loader)
            accs.append((pred == y).mean())
            missed += int(((y == 1) & (pred == 0)).sum())
            rejected += int(((y == 0) & (pred == 1)).sum())
        print(f"  {name:<6} seed {seed} | 검증 {val_acc:.3f} | 시험 " + " / ".join(f"{a:6.1%}" for a in accs)
              + f" | 불량 놓침 {missed:2d}, 정상 오판 {rejected:2d}")
