"""06-1. CNN vs ViT — 이미지를 조각내 섞거나 가렸을 때 누가 더 버티는가

Imagenette 검증셋(ImageNet의 10개 클래스, 3,925장)을 세 모델로 분류한다.
모델은 1,000개 클래스 전체 중에서 답을 고르고, 그 답이 정답 클래스와 같아야 맞은 것으로 센다.
조건: 원본 / 4x4 조각 섞기(56px 조각) / 14x14 조각 섞기(16px 조각) / 16px 조각 절반 가리기

실행 (저장소 루트에서):
    python data/download_imagenette.py
    python ch06_vit/06_1_cnn_vs_vit.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "datasets" / "imagenette2-320" / "val"
OUT = ROOT / "outputs" / "ch06"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"ResNet-50 (CNN)": "resnet50.a1_in1k",
          "DeiT-S (ViT)": "deit_small_patch16_224.fb_in1k",
          "Swin-T": "swin_tiny_patch4_window7_224.ms_in1k"}
WNID_TO_IN1K = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
                "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}


def shuffle_patches(x, grid, gen):
    """224x224 이미지를 grid x grid 조각으로 잘라 무작위로 섞는다"""
    b, c, h, w = x.shape
    s = h // grid
    p = x.reshape(b, c, grid, s, grid, s).permute(0, 2, 4, 1, 3, 5).reshape(b, grid * grid, c, s, s)
    p = torch.stack([p[i, torch.randperm(grid * grid, generator=gen)] for i in range(b)])
    return p.reshape(b, grid, grid, c, s, s).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def occlude_patches(x, ratio, gen, s=16):
    """16x16 조각 중 ratio만큼을 무작위로 골라 0(정규화 후 평균색)으로 덮는다"""
    b, c, h, w = x.shape
    keep = (torch.rand(b, 1, h // s, w // s, generator=gen) >= ratio).float()
    return x * keep.repeat_interleave(s, 2).repeat_interleave(s, 3)


CONDITIONS = {"원본": lambda x, g: x,
              "4x4 섞기": lambda x, g: shuffle_patches(x, 4, g),
              "14x14 섞기": lambda x, g: shuffle_patches(x, 14, g),
              "절반 가리기": lambda x, g: occlude_patches(x, 0.5, g)}

results = {}
for name, mid in MODELS.items():
    model = timm.create_model(mid, pretrained=True).eval().to(device)
    ds = ImageFolder(VAL, transform=create_transform(**resolve_data_config({}, model=model)))
    target_map = torch.tensor([WNID_TO_IN1K[c] for c in ds.classes])
    loader = DataLoader(ds, batch_size=100, num_workers=8)
    results[name] = {}
    t = time.perf_counter()
    for cond, fn in CONDITIONS.items():
        gen = torch.Generator().manual_seed(0)                   # 모든 모델에 같은 섞기·가리기 순서
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(fn(x, gen).to(device)).argmax(-1).cpu()
                correct += (pred == target_map[y]).sum().item()
        results[name][cond] = correct / len(ds)
    print(f"{name}: {len(ds)}장 x 조건 {len(CONDITIONS)}개, {time.perf_counter() - t:.0f}초")

print(f"\n{'모델':<17}" + "".join(f"{c:>12}" for c in CONDITIONS))
for name, r in results.items():
    print(f"{name:<17}" + "".join(f"{v:>12.1%}" for v in r.values()))

# 그림 — bus.jpg에 네 조건을 적용한 모습 (정규화를 되돌려 저장, 가린 곳은 평균색)
cfg = resolve_data_config({}, model=model)
x = create_transform(**cfg)(Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB"))[None]
gen = torch.Generator().manual_seed(0)
mean, std = torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
tiles = [((fn(x, gen)[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) for fn in CONDITIONS.values()]
cv2.imwrite(str(OUT / "06_1_conditions.jpg"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
