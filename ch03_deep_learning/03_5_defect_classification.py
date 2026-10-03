"""03-5. 실습: 정상/불량 이미지 분류 — ResNet18 미세조정, 평가 지표, 02-6 Rule 검사와 비교

실행 (저장소 루트에서, 03-3의 데이터 준비 후):
    python ch03_deep_learning/03_5_defect_classification.py           # 학습 + 평가 + 속도
    python ch03_deep_learning/03_5_defect_classification.py --bench   # 저장된 모델로 속도만 (GPU 없는 PC용)
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from parts_cls import (DEFECTS, LIGHTS, MEAN, STD, ROOT, TEST_DIR, TRAIN_DIR, labels_of, make_loader, predict,
                       read_rows, seed_everything, split_rows, train)

ap = argparse.ArgumentParser()
ap.add_argument("--bench", action="store_true", help="학습 없이 저장된 모델로 속도만 잰다")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "outputs" / "ch03"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = OUT / "resnet18_parts.pt"
norm = transforms.Normalize(MEAN, STD)
flip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), norm])


def resnet18(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def scores(y, pred):
    """불량(1)을 찾는 문제로 보고 Precision·Recall·F1을 계산한다"""
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "acc": (tp + tn) / len(y),
            "precision": precision, "recall": recall, "f1": f1}


test_rows = read_rows(TEST_DIR)
test_loader = make_loader(TEST_DIR, test_rows, norm)
y = labels_of(test_loader)


# 5. 속도 — 학습된 모델로 추론 (맨 아래에서 실행)
def bench():
    model = resnet18(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    x1 = next(iter(test_loader))[0][:1]
    batch = torch.cat([x for x, _ in test_loader])
    print(f"\n[5] 추론 속도 (모델 파일 {WEIGHTS.stat().st_size / 1024 ** 2:.1f}MB, 입력 {tuple(x1.shape[1:])})")
    with torch.no_grad():
        for dev in (["cuda", "cpu"] if device == "cuda" else ["cpu"]):
            m = model.to(dev)
            for _ in range(3):
                m(x1.to(dev))                               # 워밍업
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            t = time.perf_counter()
            for _ in range(20):
                m(x1.to(dev))
            sync()
            one = (time.perf_counter() - t) / 20 * 1000
            t = time.perf_counter()
            m(batch.to(dev))
            sync()
            per = (time.perf_counter() - t) / len(batch) * 1000
            print(f"  {dev:<4} (스레드 {torch.get_num_threads()}): 1장씩 {one:6.1f} ms/장 | {len(batch)}장 한 번에 {per:6.2f} ms/장")


if args.bench:            # --bench: 학습 없이 저장된 모델로 속도만 잰다
    bench()
    raise SystemExit

# 1. 학습 — 세 조명 720장, 뒤집기 변형, 10에폭
rows = read_rows(TRAIN_DIR)
train_rows, val_rows = split_rows(rows)
seed_everything(0)
model = resnet18()
print("[1] ResNet18 미세조정")
val_acc, sec = train(model, make_loader(TRAIN_DIR, train_rows, flip, shuffle=True),
                     make_loader(TRAIN_DIR, val_rows, norm), epochs=10, lr=1e-4, device=device)
print(f"  → 검증 정확도 {val_acc:.3f}, 학습 {sec:.1f}초")
torch.save(model.state_dict(), WEIGHTS)

# 2. 평가 지표 — 02-6과 같은 시험 180장
pred, prob = predict(model, test_loader, device)
s = scores(y, pred)
print("\n[2] 시험 180장 혼동 행렬 (행: 정답, 열: 예측)")
print(f"           예측 정상  예측 불량")
print(f"  정답 정상 {s['tn']:8d} {s['fp']:9d}")
print(f"  정답 불량 {s['fn']:8d} {s['tp']:9d}")
print(f"  Accuracy {s['acc']:.3f} | Precision {s['precision']:.3f} | Recall {s['recall']:.3f} | F1 {s['f1']:.3f}")
for light in LIGHTS:
    idx = np.array([r["light"] == light for r in test_rows])
    ls = scores(y[idx], pred[idx])
    print(f"  {light:<9} Accuracy {ls['acc']:6.1%} | 불량 놓침 {ls['fn']:2d} | 정상 오판 {ls['fp']:2d}")
print("  불량 유형별 Recall:", " ".join(
    f"{d}={sum(p for p, r in zip(pred, test_rows) if r['label'] == d)}/15" for d in DEFECTS))

# 3. 임계값 — "불량일 확률"이 얼마 이상이면 불량으로 볼까
print("\n[3] 불량 확률 분포와 임계값")
print(f"  정상품의 불량 확률 최댓값 {prob[y == 0].max():.4f} | 불량품의 불량 확률 최솟값 {prob[y == 1].min():.4f}")
for th in [0.5, 0.9, 0.99, 0.999]:
    ts = scores(y, (prob >= th).astype(int))
    print(f"  임계값 {th:<6} → 불량 놓침 {ts['fn']:2d} | 정상 오판 {ts['fp']:2d} | Precision {ts['precision']:.3f} | Recall {ts['recall']:.3f}")

# 4. 학습 데이터에 없던 불량 — chip을 빼고 학습하면?
seed_everything(0)
no_chip = resnet18()
train(no_chip, make_loader(TRAIN_DIR, [r for r in train_rows if r["label"] != "chip"], flip, shuffle=True),
      make_loader(TRAIN_DIR, [r for r in val_rows if r["label"] != "chip"], norm),
      epochs=10, lr=1e-4, device=device, log=False)
pred2, prob2 = predict(no_chip, test_loader, device)
chip_idx = np.array([r["label"] == "chip" for r in test_rows])
print("\n[4] chip 없이 학습한 모델")
print(f"  chip 검출 {int(pred2[chip_idx].sum())}/15 | chip의 불량 확률 {[round(float(p), 2) for p in prob2[chip_idx]]}")
s2 = scores(y[~chip_idx], pred2[~chip_idx])
print(f"  chip을 뺀 나머지 165장: Accuracy {s2['acc']:.3f}, 불량 놓침 {s2['fn']}, 정상 오판 {s2['fp']}")

bench()
