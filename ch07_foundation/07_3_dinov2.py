"""07-3. DINOv2 — 라벨 없이 배운 특징으로 분류하고, 물체를 찾기

1) DINOv2의 출력 — CLS 토큰과 조각(Patch) 토큰
2) 특징을 고정하고 분류 — DINOv2-S, CLIP ViT-B/32, ImageNet 지도학습 ResNet-50을 같은 방법으로 비교
   a. Imagenette(ImageNet의 10종): kNN, 클래스당 1·5·10장만 쓴 분류, 선형 분류기
   b. 03장의 합성 부품(정상/불량): ImageNet과 거리가 먼 문제에서도 통하는가
3) CLS Attention과 조각 특징의 주성분(PCA) — 06-1의 지도학습 ViT(DeiT-S)와 같은 크롭·같은 물체 마스크로 비교

실행 (저장소 루트에서, data/download_imagenette.py와 03장의 data/make_parts.py 실행 후):
    python ch07_foundation/07_3_dinov2.py
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import open_clip
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets" / "imagenette2-320"
OUT = ROOT / "outputs" / "ch07"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 1. DINOv2의 출력
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")      # 짧은 변 256으로 줄이고 가운데 224 자르기
dino = AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="eager").eval().to(device)
img = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")
with torch.no_grad():
    out = dino(**proc(images=img, return_tensors="pt").to(device))
print(f"DINOv2-S 파라미터 {sum(p.numel() for p in dino.parameters()):,} | 조각 크기 {dino.config.patch_size}px")
print(f"  last_hidden_state {tuple(out.last_hidden_state.shape)} = CLS 1 + 조각 16x16=256, 384차원")
print(f"  pooler_output {tuple(out.pooler_output.shape)} (CLS 토큰, 이미지 전체를 대표하는 벡터)")

# 2. 특징 고정 분류 — 세 모델의 특징을 뽑아 같은 방법으로 분류
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def hf_extractor(name):
    p = AutoImageProcessor.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval().to(device)
    return (lambda im: p(images=im, return_tensors="pt")["pixel_values"][0]), (lambda x: m(pixel_values=x).pooler_output), IMAGENET_MEAN, IMAGENET_STD


def clip_extractor():
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    return pre, m.eval().encode_image, open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD


def timm_extractor(name):
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)   # 분류 머리를 뗀 2048차원 특징
    return create_transform(**resolve_data_config({}, model=m)), m, IMAGENET_MEAN, IMAGENET_STD


class Parts(Dataset):
    """03장의 합성 부품 — 640x480 흑백을 224x224로 줄인다 (가운데만 자르면 가장자리 불량이 잘릴 수 있어 통째로)"""

    def __init__(self, folder, transform):
        with open(folder / "labels.csv", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.folder, self.transform = folder, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(Image.open(self.folder / r["file"]).convert("RGB")), int(r["label"] != "ok")


def extract(ds, fn):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=100, num_workers=8):
            f = fn(x.to(device)).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
            labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


def knn(tr, ytr, te, k, n_cls):
    top = (te @ tr.T).topk(k, dim=1)                            # 코사인 유사도가 높은 k장이 유사도만큼 투표
    votes = torch.zeros(len(te), n_cls, device=device).scatter_add_(1, ytr[top.indices], top.values)
    return votes.argmax(1)


def linear_probe(tr, ytr, te, n_cls, epochs=100):
    clf = torch.nn.Linear(tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):                                     # 특징은 고정, 선형층 하나만 학습 (전체 배치)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(clf(tr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return clf(te).argmax(1)


EXTRACTORS = {"DINOv2-S (라벨 없이)": lambda: hf_extractor("facebook/dinov2-small"),
              "CLIP ViT-B/32 (글과 함께)": clip_extractor,
              "ResNet-50 (ImageNet 라벨)": lambda: timm_extractor("resnet50.a1_in1k")}
results = {}
for name, make in EXTRACTORS.items():
    transform, fn, mean, std = make()
    t0 = time.perf_counter()
    tr, ytr = extract(ImageFolder(DATA / "train", transform=transform), fn)
    te, yte = extract(ImageFolder(DATA / "val", transform=transform), fn)
    acc = lambda pred: (pred == yte).float().mean().item()
    few = []
    for n in (1, 5, 10):                                       # 클래스마다 n장만 라벨이 있다고 가정 (5번 반복 평균)
        runs = []
        for seed in range(5):
            g = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.cat([torch.nonzero(ytr == c).flatten()[torch.randperm(int((ytr == c).sum()), generator=g)[:n].to(device)] for c in range(10)])
            runs.append(acc(knn(tr[idx], ytr[idx], te, min(n, 5), 10)))
        few.append(np.mean(runs))
    row = [tr.shape[1], *few, acc(knn(tr, ytr, te, 20, 10)), acc(linear_probe(tr, ytr, te, 10))]
    parts_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    ptr, pytr = extract(Parts(ROOT / "data" / "parts_train", parts_tf), fn)
    pte, pyte = extract(Parts(ROOT / "data" / "parts", parts_tf), fn)
    pacc = lambda pred: (pred == pyte).float().mean().item()
    row += [pacc(knn(ptr, pytr, pte, 20, 2)), pacc(linear_probe(ptr, pytr, pte, 2)), time.perf_counter() - t0]
    results[name] = row
    print(f"{name}: Imagenette 학습 {len(ytr):,}장·평가 {len(yte):,}장, 부품 학습 {len(pytr)}장·시험 {len(pyte)}장")

print(f"\n{'특징':<26}{'차원':>6} | Imagenette 10종: {'1장':>6}{'5장':>7}{'10장':>7}{'kNN':>7}{'선형':>7} | 부품 정상/불량: {'kNN':>6}{'선형':>7} | {'시간(초)'}")
for name, r in results.items():
    print(f"{name:<26}{r[0]:>6} |                  {r[1]:>6.1%}{r[2]:>7.1%}{r[3]:>7.1%}{r[4]:>7.1%}{r[5]:>7.1%} |                {r[6]:>6.1%}{r[7]:>7.1%} | {r[8]:.0f}")

# 3. 어디를 보는가 — CLS Attention이 물체 위에 있는 비율, 조각 특징의 주성분
seg = YOLO("yolo26s-seg.pt")
models = {"DeiT-S (06-1, 지도학습)": AutoModel.from_pretrained("facebook/deit-small-patch16-224", attn_implementation="eager"),
          "DINOv2-S": dino,
          "DINOv2-S + Registers": AutoModel.from_pretrained("facebook/dinov2-with-registers-small", attn_implementation="eager")}
models = {k: m.eval().to(device) for k, m in models.items()}
rows = []
for path in [ROOT / "data" / "images" / "bus.jpg", ROOT / "data" / "images" / "coco_cats.jpg"]:
    pix = proc(images=Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"].to(device)   # 세 모델에 같은 224x224
    crop = (pix[0].permute(1, 2, 0).cpu().numpy() * np.array(proc.image_std) + np.array(proc.image_mean)).clip(0, 1)
    crop = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)              # 모델이 실제로 본 224x224
    r = seg(cv2.resize(crop, (640, 640)), classes=[0, 2, 5, 7, 15], verbose=False)[0]     # 사람·차·버스·트럭·고양이
    mask = (r.masks.data.sum(0) > 0).float().cpu().numpy()
    panel = [crop]
    for name, m in models.items():
        with torch.no_grad():
            o = m(pixel_values=pix, output_attentions=True)
        n_reg = getattr(m.config, "num_register_tokens", 0)
        att = o.attentions[-1][0, :, 0, 1 + n_reg:].mean(0)                                # CLS → 조각들, 헤드 평균
        grid = int(len(att) ** 0.5)                                                         # DeiT 14x14, DINOv2 16x16
        att = (att / att.sum()).reshape(grid, grid).cpu().numpy()
        obj = cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA)                  # 조각마다 물체가 차지하는 비율
        top = np.sort(att.ravel())[::-1]
        print(f"[{path.stem}] {name:<24}: 물체 위 Attention {float((att * obj).sum()):.1%} (물체 면적 {obj.mean():.1%}), "
              f"상위 10% 조각에 {top[:max(1, grid * grid // 10)].sum():.1%}")
        heat = cv2.applyColorMap((cv2.resize(att / att.max(), (224, 224), interpolation=cv2.INTER_CUBIC).clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panel.append(cv2.addWeighted(crop, 0.45, heat, 0.55, 0))
    # DINOv2는 입력 크기를 바꿀 수 있다 — 448x448(조각 32x32)로 조각 특징을 뽑아 주성분 3개를 RGB로
    pix448 = proc(images=Image.open(path).convert("RGB"), size={"shortest_edge": 512},
                  crop_size={"height": 448, "width": 448}, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        patches = dino(pixel_values=pix448).last_hidden_state[0, 1:].float()                # 1024 x 384
    patches = patches - patches.mean(0)
    _, _, v = torch.pca_lowrank(patches, q=3)
    pc = (patches @ v[:, :3]).cpu().numpy()
    pc = (pc - pc.min(0)) / (pc.max(0) - pc.min(0))
    panel.append(cv2.resize((pc.reshape(32, 32, 3) * 255).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST))
    rows.append(np.hstack(panel))
cv2.imwrite(str(OUT / "07_3_dinov2_attention_pca.jpg"), np.vstack(rows))
print(f"→ {OUT / '07_3_dinov2_attention_pca.jpg'} (원본 | DeiT-S | DINOv2-S | DINOv2-S+Registers Attention | DINOv2 조각 특징 PCA)")
