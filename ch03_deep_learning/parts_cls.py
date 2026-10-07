"""03장 공통 모듈 — 합성 부품 이미지 Dataset, 학습·평가 함수

03-3, 03-4, 03-5 예제가 이 파일을 가져다 쓴다 (같은 폴더에 있어야 한다).

    학습·검증 데이터: data/parts_train (python data/make_parts.py --n 300 --seed 2027 --out data/parts_train)
    시험 데이터    : data/parts       (02장과 같은 180장, seed 2026) → 02-6 Rule 검사와 같은 문제로 비교
"""
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "parts_train"
TEST_DIR = ROOT / "data" / "parts"
LIGHTS = ["normal", "dim", "gradient"]
DEFECTS = ["missing_hole", "chip", "crack", "rotated", "offset", "small"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]      # ImageNet 평균·표준편차


def seed_everything(seed=0):
    """결과를 되도록 같게 만들기 위해 난수 시드를 고정한다 (GPU 종류·드라이버가 다르면 숫자가 조금 달라질 수 있다)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True      # 매번 같은 연산 방식을 쓰도록 고정 (조금 느려질 수 있다)
    torch.backends.cudnn.benchmark = False


def read_rows(folder, lights=LIGHTS):
    with open(Path(folder) / "labels.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["light"] in lights]


def split_rows(rows, val_every=5):
    """5장마다 1장을 검증용으로 뺀다 (조명·불량 종류가 고르게 나뉜다)"""
    train = [r for i, r in enumerate(rows) if i % val_every != 0]
    val = [r for i, r in enumerate(rows) if i % val_every == 0]
    return train, val


class PartsDataset(Dataset):
    """이미지 한 장 → (3 x 168 x 224 텐서, 라벨 0=정상·1=불량)"""

    def __init__(self, folder, rows, transform=None, size=(224, 168)):
        self.folder, self.rows, self.transform, self.size = Path(folder), rows, transform, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.folder / r["file"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(255).unsqueeze(0).repeat(3, 1, 1)   # 흑백 1채널 → 3채널
        if self.transform:
            x = self.transform(x)
        return x, 0 if r["label"] == "ok" else 1


def make_loader(folder, rows, transform=None, shuffle=False, batch_size=32):
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(PartsDataset(folder, rows, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0 if sys.platform == "win32" else 4, generator=g)   # Windows는 작업 프로세스 없이


def train(model, train_loader, val_loader, epochs, lr, device, log=True):
    """학습 루프. 에폭마다 검증 정확도를 보고, 가장 좋았던 가중치로 되돌린다"""
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val_acc = (predict(model, val_loader, device)[0] == labels_of(val_loader)).mean()
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log:
            print(f"  epoch {epoch:2d} | train loss {total / len(train_loader.dataset):.4f} | val acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    return best_acc, time.perf_counter() - t0


@torch.no_grad()
def predict(model, loader, device):
    """(예측 라벨, 불량일 확률) — 확률이 0.5 이상이면 불량"""
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    return (p >= 0.5).astype(int), p


def labels_of(loader):
    return np.array([0 if r["label"] == "ok" else 1 for r in loader.dataset.rows])


def test_report(model, device, transform=None, title=""):
    """02-6과 같은 시험 데이터(조명별 60장)로 정확도와 불량 유형별 검출 수를 출력한다"""
    if title:
        print(title)
    for light in LIGHTS:
        rows = read_rows(TEST_DIR, [light])
        pred, _ = predict(model, make_loader(TEST_DIR, rows, transform), device)
        y = np.array([0 if r["label"] == "ok" else 1 for r in rows])
        caught = {d: int(sum(p for p, r in zip(pred, rows) if r["label"] == d)) for d in DEFECTS}
        missed, false_reject = int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
        print(f"  {light:<9} 정확도 {(pred == y).mean():6.1%} | 불량 놓침 {missed:2d} | 정상 오판 {false_reject:2d} | "
              + " ".join(f"{d}={c}/5" for d, c in caught.items()))
