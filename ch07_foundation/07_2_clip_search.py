"""07-2. 실습: CLIP 유사 이미지 검색 — COCO val2017 500장

1) 이미지 500장의 Embedding을 뽑아 faiss 색인에 넣는다
2) 사람이 쓴 설명 문장(이미지마다 5개, 2,502개)으로 검색해, 원래 이미지를 몇 위에 찾는지 잰다 (Recall@K)
3) 이미지로 이미지 검색 — 비슷한 사진 찾기
4) 질의 유형별로 잘 되는 것과 안 되는 것 — 정답 박스로 자동 채점

실행 (저장소 루트에서, data/download_coco_val.py 실행 후):
    python ch07_foundation/07_2_clip_search.py
    python ch07_foundation/07_2_clip_search.py --query "a dog playing with a frisbee"
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ViT-B-32")
ap.add_argument("--pretrained", default="openai")
ap.add_argument("--query", default="", help="직접 검색해 볼 문장")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
tokenizer = open_clip.get_tokenizer(args.model)
model.eval()


def embed_text(texts, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):                   # 문장이 많으면 나눠서 (GPU 메모리)
            t = model.encode_text(tokenizer(texts[i:i + batch]).to(device)).float()
            out.append((t / t.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


# 1. 이미지 Embedding → faiss 색인 (길이 1로 맞춘 벡터의 내적 = 코사인 유사도)
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
caps = json.loads((COCO_DIR / "captions.json").read_text(encoding="utf-8"))
images = inst["images"]
t0 = time.perf_counter()
feats = []
with torch.no_grad():
    for i in range(0, len(images), 50):
        x = torch.stack([preprocess(Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")) for im in images[i:i + 50]])
        f = model.encode_image(x.to(device)).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
feats = np.concatenate(feats).astype("float32")
index = faiss.IndexFlatIP(feats.shape[1])                       # 전부 비교하는 가장 단순한 색인
index.add(feats)
print(f"{args.model} ({args.pretrained}) | 이미지 {index.ntotal}장 색인, {feats.shape[1]}차원, {time.perf_counter() - t0:.1f}초")

# 2. 설명 문장으로 원래 이미지 찾기 — Recall@K
pos = {im["id"]: i for i, im in enumerate(images)}
queries = [(c["caption"].strip(), pos[c["image_id"]]) for c in caps["annotations"]]
q = embed_text([c for c, _ in queries])
t0 = time.perf_counter()
_, top = index.search(q, 10)
ms = (time.perf_counter() - t0) / len(queries) * 1000
rank_hit = np.array([[gt in row[:k] for k in (1, 5, 10)] for (_, gt), row in zip(queries, top)])
print(f"설명 문장 {len(queries)}개로 검색 (질의 하나 {ms:.3f} ms)")
print("  Recall@1 {:.1%} | Recall@5 {:.1%} | Recall@10 {:.1%}".format(*rank_hit.mean(0)))
miss = [(c, gt) for (c, gt), h in zip(queries, rank_hit) if not h[2]]
print(f"  10위 안에 원래 이미지가 없는 문장 {len(miss)}개, 예: " + " / ".join(f'"{c}"' for c, _ in miss[:3]))

# 3. 이미지로 이미지 검색 — 자기 자신을 뺀 가장 비슷한 3장
cats = {c["id"]: c["name"] for c in inst["categories"]}
objs = defaultdict(Counter)                                     # 이미지마다 정답 물체 개수
for a in inst["annotations"]:
    objs[a["image_id"]][cats[a["category_id"]]] += 1
sims, nn = index.search(feats[:3], 4)
print("\n이미지로 검색 (앞 3장)")
for i in range(3):
    main = ", ".join(n for n, _ in objs[images[i]["id"]].most_common(3))
    print(f"  {images[i]['file_name']} [{main}]")
    for s, j in zip(sims[i][1:], nn[i][1:]):
        print(f"    {s:.3f} {images[j]['file_name']} [{', '.join(n for n, _ in objs[images[j]['id']].most_common(3))}]")

# 4. 질의 유형별 정밀도 — 상위 5장 중 조건을 만족하는 비율 (정답 박스로 판정)
TESTS = {
    "물체": [("a photo of a giraffe", lambda o: o["giraffe"] > 0), ("a photo of a pizza", lambda o: o["pizza"] > 0),
            ("a photo of a train", lambda o: o["train"] > 0), ("a photo of a laptop", lambda o: o["laptop"] > 0)],
    "개수": [("a photo of exactly one person", lambda o: o["person"] == 1), ("a photo of three people", lambda o: o["person"] == 3),
            ("a photo of two zebras", lambda o: o["zebra"] == 2), ("a photo of five or more cars", lambda o: o["car"] >= 5)],
    "부정": [("a street with no cars", lambda o: o["car"] == 0), ("a room without people", lambda o: o["person"] == 0),
            ("a table with no food on it", lambda o: o["dining table"] > 0 and not any(o[f] for f in ["pizza", "cake", "sandwich", "donut", "bowl"])),
            ("an animal that is not a dog", lambda o: o["dog"] == 0 and any(o[a] for a in ["cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird"]))],
    "두 물체 함께": [("a dog and a cat together", lambda o: o["dog"] > 0 and o["cat"] > 0), ("a person riding a horse", lambda o: o["person"] > 0 and o["horse"] > 0),
                 ("a laptop next to a cup", lambda o: o["laptop"] > 0 and o["cup"] > 0), ("a person holding an umbrella", lambda o: o["person"] > 0 and o["umbrella"] > 0)],
}
print("\n질의 유형별 상위 5장 정밀도 (정답 박스로 판정, 괄호는 500장 전체에서 조건을 만족하는 비율)")
for kind, tests in TESTS.items():
    scores = []
    for text, cond in tests:
        _, top5 = index.search(embed_text([text]), 5)
        hit = sum(cond(objs[images[j]["id"]]) for j in top5[0])
        base = np.mean([cond(objs[im["id"]]) for im in images])
        scores.append(hit / 5)
        print(f"  [{kind}] {text:<34} {hit}/5 (전체 {base:.1%})")
    print(f"  → {kind} 평균 {np.mean(scores):.0%}")

if args.query:
    s, top5 = index.search(embed_text([args.query]), 5)
    print(f"\n'{args.query}'")
    for v, j in zip(s[0], top5[0]):
        print(f"  {v:.3f} {images[j]['file_name']} [{', '.join(f'{n} {c}' for n, c in objs[images[j]['id']].most_common(4))}]")
