"""16-3 프로젝트: 자연어 이미지 검색·분할 시스템 — CLIP + Vector DB(faiss) → Grounding DINO + SAM 2.1

1) 색인: 이미지 500장(COCO val2017)을 CLIP ViT-L-14로 Embedding해 faiss 색인을 디스크에 저장 (있으면 다시 쓴다)
2) 질의: "검색 문장"으로 비슷한 이미지 5장을 찾고, 그 안에서 "찾을 대상"을 Grounding DINO 박스 → SAM 2.1 마스크로 칠한다
3) 평가: 정답 이미지가 5장 이상인 COCO 클래스마다 "a photo of a {클래스}"로 검색
   - 검색: 5장 중 그 물체가 실제로 있는 사진의 비율 (Precision@5)
   - 분할: 찾은 사진 안의 정답 마스크를 얼마나 찾았나 (마스크 IoU 0.5)
   - 시간: 단계별

실행 (저장소 루트에서):
    python ch16_projects/p3_text_search_segment/search_segment.py
    python ch16_projects/p3_text_search_segment/search_segment.py --query "a cat sleeping on a couch" --find cat
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parents[2]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch16" / "p3"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--query", default="", help="검색 문장 (영어)")
ap.add_argument("--find", default="", help="찾은 이미지 안에서 칠할 대상 (영어 명사)")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()
device = "cuda"
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
images = inst["images"]

# 1. 색인 — 한 번 만들어 디스크에 저장하고, 다음부터는 불러온다
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer("ViT-L-14")
clip.eval()
index_path = OUT / "coco500_vitl14.faiss"
if index_path.exists():
    index = faiss.read_index(str(index_path))
    print(f"색인 불러오기: {index.ntotal}장")
else:
    t = time.perf_counter()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 50):
            x = torch.stack([preprocess(Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")) for im in images[i:i + 50]])
            f = clip.encode_image(x.to(device)).float()
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    index = faiss.IndexFlatIP(768)                                            # 길이 1 벡터의 내적 = 코사인 유사도 (07-2)
    index.add(np.concatenate(feats).astype("float32"))
    faiss.write_index(index, str(index_path))
    print(f"색인 만들기: {index.ntotal}장, {time.perf_counter() - t:.1f}초 → {index_path.name} ({index_path.stat().st_size / 1e6:.1f} MB)")

gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval().to(device)
sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus").eval().to(device)


def search(text, k):
    with torch.no_grad():
        q = clip.encode_text(tokenizer([text]).to(device)).float()
    q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy()
    scores, idx = index.search(q, k)
    return [(images[i], float(s)) for i, s in zip(idx[0], scores[0])]


def segment(image, target):
    """09-3과 같은 조합: 글 → Grounding DINO 박스(0.35) → SAM 2.1 마스크"""
    inp = gd_proc(images=image, text=f"{target}.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.35, text_threshold=0.25,
                                                       target_sizes=[image.size[::-1]])[0]
    if len(r["boxes"]) == 0:
        return []
    s_inp = sam_proc(images=image, input_boxes=[r["boxes"].tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        o = sam(**s_inp, multimask_output=False)
    return list(sam_proc.post_process_masks(o.pred_masks.cpu(), s_inp["original_sizes"].cpu())[0][:, 0].numpy())


def run(text, target, k):
    """질의 하나: 검색 → 분할. 단계별 시간도 돌려준다"""
    t0 = time.perf_counter()
    hits = search(text, k)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    results = [(im, s, segment(Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB"), target)) for im, s in hits]
    torch.cuda.synchronize()
    return results, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000


run("a photo", "person", 1)                                                   # 워밍업
if args.query:
    res, ms_s, ms_g = run(args.query, args.find, args.k)
    print(f"'{args.query}' → '{args.find}': 검색 {ms_s:.0f}ms, 분할 {ms_g:.0f}ms")
    for im, s, masks in res:
        print(f"  {im['file_name']} 유사도 {s:.3f}, '{args.find}' {len(masks)}개 ({sum(int(m.sum()) for m in masks):,}화소)")
    raise SystemExit

# 2. 몇 가지 문장으로 써 보기
names = {c["id"]: c["name"] for c in inst["categories"]}
present = defaultdict(set)
gt_masks = defaultdict(lambda: defaultdict(list))
for a in inst["annotations"]:
    present[a["image_id"]].add(names[a["category_id"]])
    if not a["iscrowd"]:
        gt_masks[a["image_id"]][names[a["category_id"]]].append(a)
for text, target in [("a cat sleeping on a couch", "cat"), ("a man riding a wave on a surfboard", "surfboard"),
                     ("a plate of food with broccoli", "broccoli")]:
    res, ms_s, ms_g = run(text, target, 3)
    print(f"'{text}' → '{target}': 검색 {ms_s:.0f}ms, 분할 {ms_g:.0f}ms (3장)")
    for im, s, masks in res:
        print(f"  #{im['id']} 유사도 {s:.3f} | 정답에 '{target}' {'있음' if target in present[im['id']] else '없음'} | 칠한 마스크 {len(masks)}개")

# 3. 평가 — 클래스마다 "a photo of a {클래스}"
count = defaultdict(int)
for iid, s in present.items():
    for c in s:
        count[c] += 1
classes = sorted(c for c, n in count.items() if n >= 5)
p_at_k, s_stats, times = [], defaultdict(int), defaultdict(list)
for cls in classes:
    res, ms_s, ms_g = run(f"a photo of a {cls}", cls, args.k)
    times["검색"].append(ms_s)
    times["분할"].append(ms_g / args.k)
    p_at_k.append(np.mean([cls in present[im["id"]] for im, _, _ in res]))
    for im, _, masks in res:
        gts = [mask_utils.decode(mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])).max(axis=2)
               if isinstance(a["segmentation"], list) else mask_utils.decode(a["segmentation"]) for a in gt_masks[im["id"]][cls]]
        used = set()
        hit = 0
        for m in masks:
            best, bj = 0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                iou = np.logical_and(m, g).sum() / max(1, np.logical_or(m, g).sum())
                if iou > best:
                    best, bj = iou, j
            if best >= 0.5:
                used.add(bj)
                hit += 1
        s_stats["정답"] += len(gts)
        s_stats["예측"] += len(masks)
        s_stats["맞힘"] += hit
print(f"\n평가: 정답 이미지가 5장 이상인 COCO 클래스 {len(classes)}개, 클래스마다 \"a photo of a ...\"로 {args.k}장 검색")
print(f"  검색 Precision@{args.k} (그 물체가 정말 있는 사진의 비율) {np.mean(p_at_k):.1%}, 5장 모두 맞힌 클래스 {np.mean([p == 1 for p in p_at_k]):.0%}")
print(f"  찾은 사진 안의 분할: 정답 마스크 {s_stats['정답']}개 중 {s_stats['맞힘']}개 (재현율 {s_stats['맞힘'] / s_stats['정답']:.1%}), "
      f"칠한 마스크 {s_stats['예측']}개 중 맞은 것 {s_stats['맞힘'] / max(1, s_stats['예측']):.1%}")
print(f"  시간: 검색 {np.median(times['검색']):.0f}ms (질의당), 분할 {np.median(times['분할']):.0f}ms (사진당) → 질의 하나에 약 "
      f"{(np.median(times['검색']) + args.k * np.median(times['분할'])) / 1000:.1f}초")
worst = sorted(zip(p_at_k, classes))[:5]
print("  검색이 가장 약했던 클래스: " + ", ".join(f"{c} {p:.0%}" for p, c in worst))
