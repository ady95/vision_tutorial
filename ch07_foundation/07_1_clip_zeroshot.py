"""07-1. CLIP — 이미지와 글을 같은 공간에 놓기: 유사도와 Zero-shot 분류

1) 이미지·글 Embedding의 모양과 코사인 유사도
2) Imagenette 검증셋 3,925장을 학습 없이 분류 — 글로 쓴 클래스 이름만으로
   (10종 중에서 / 1,000종 중에서 — 06-1의 지도학습 모델과 같은 조건)

실행 (저장소 루트에서, data/download_imagenette.py 실행 후):
    python ch07_foundation/07_1_clip_zeroshot.py
"""
import time
from collections import Counter
from pathlib import Path

import open_clip
import torch
from PIL import Image
from timm.data import ImageNetInfo
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "datasets" / "imagenette2-320" / "val"
device = "cuda" if torch.cuda.is_available() else "cpu"
WNID_TO_IN1K = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
                "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}
IN1K_NAMES = [ImageNetInfo().index_to_description(i).split(",")[0] for i in range(1000)]   # 클래스 이름 (첫 번째 이름)

# 1. 이미지와 글을 같은 공간으로 — ViT-B/32 (OpenAI 공개 가중치)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()
images = [ROOT / "data" / "images" / "bus.jpg", ROOT / "data" / "images" / "coco_cats.jpg"]
texts = ["a photo of a bus", "a photo of cats", "a photo of a dog", "people walking on a street",
         "two cats sleeping on a pink blanket", "a red sports car"]
with torch.no_grad():
    img_emb = model.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in images]).to(device))
    txt_emb = model.encode_text(tokenizer(texts).to(device))
print(f"이미지 Embedding {tuple(img_emb.shape)}, 글 Embedding {tuple(txt_emb.shape)} — 같은 512차원 공간")
img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)          # 길이를 1로 맞추면 내적 = 코사인 유사도
txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
sim = img_emb @ txt_emb.T
print(f"\n{'':<36}" + "".join(f"{p.stem:>12}" for p in images))
for j, t in enumerate(texts):
    print(f"{t:<36}" + "".join(f"{float(sim[i, j]):>12.3f}" for i in range(len(images))))
prob = (model.logit_scale.exp().item() * sim).softmax(-1)       # 학습 때 쓴 온도(100배)로 키운 뒤 Softmax
for i, p in enumerate(images):
    print(f"{p.stem}: 가장 가까운 글 '{texts[int(prob[i].argmax())]}' (확률 {float(prob[i].max()):.2f})")

# 2. Zero-shot 분류 — 클래스 이름을 글로 써서 Embedding을 만들고, 가장 가까운 글을 답으로
for name, pretrained in [("ViT-B-32", "openai"), ("ViT-L-14", "openai")]:
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(name)
    ds = ImageFolder(VAL, transform=preprocess)
    target = torch.tensor([WNID_TO_IN1K[c] for c in ds.classes])
    t0 = time.perf_counter()
    with torch.no_grad():
        feats, labels = [], []
        for x, y in DataLoader(ds, batch_size=100, num_workers=8):
            f = model.encode_image(x.to(device))
            feats.append(f / f.norm(dim=-1, keepdim=True))
            labels.append(target[y])
        feats, labels = torch.cat(feats), torch.cat(labels).to(device)

        def classify(class_ids, template):
            t = model.encode_text(tokenizer([template.format(IN1K_NAMES[c]) for c in class_ids]).to(device))
            t = t / t.norm(dim=-1, keepdim=True)
            pred = torch.tensor(class_ids, device=device)[(feats @ t.T).argmax(-1)]
            return (pred == labels).float().mean().item(), pred

        ten = list(WNID_TO_IN1K.values())
        acc = {"10종, 이름만": classify(ten, "{}")[0],
               "10종, 'a photo of a {}.'": classify(ten, "a photo of a {}.")[0]}
        acc["1000종, 'a photo of a {}.'"], pred = classify(list(range(1000)), "a photo of a {}.")
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{name} ({pretrained}) 파라미터 {params:,} | {len(ds)}장 {time.perf_counter() - t0:.0f}초")
    for k, v in acc.items():
        print(f"  {k:<28}: {v:.1%}")
    wrong = Counter((IN1K_NAMES[int(a)], IN1K_NAMES[int(b)]) for a, b in zip(labels, pred) if a != b)
    print("  1000종에서 가장 많이 틀린 경우: " + " | ".join(f"{a} → {b} {n}장" for (a, b), n in wrong.most_common(4)))
