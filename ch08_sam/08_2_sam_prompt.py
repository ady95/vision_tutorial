"""08-2. 실습: Point·Box Prompt와 자동 마스크 생성

1) 점 하나, 박스 하나로 분할하기 — 점 하나에 마스크 후보 3개가 나오는 이유
2) 무거운 Image Encoder는 한 번, 가벼운 Mask Decoder는 Prompt마다 — 시간 나누어 재기
3) COCO val2017 500장의 정답 마스크로 채점 — 정답 박스를 주면, 점 하나를 주면 얼마나 맞나
   (SAM ViT-B, SAM 2.1 Hiera-B+, SAM 2.1 Hiera-T)
4) 자동 마스크 생성 — Prompt 없이 이미지의 모든 것을 분할

실행 (저장소 루트에서, data/download_coco_val.py 실행 후):
    python ch08_sam/08_2_sam_prompt.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import Sam2Model, Sam2Processor, SamModel, SamProcessor, pipeline

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch08"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"SAM ViT-B": ("facebook/sam-vit-base", SamModel, SamProcessor),
          "SAM 2.1 Hiera-B+": ("facebook/sam2.1-hiera-base-plus", Sam2Model, Sam2Processor),
          "SAM 2.1 Hiera-T": ("facebook/sam2.1-hiera-tiny", Sam2Model, Sam2Processor)}


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def segment(model, proc, image, emb=None, points=None, boxes=None, multimask=False):
    """이미지 하나에 Prompt 여러 개 (points: [[x, y], ...] 물체마다 점 하나 / boxes: [[x1, y1, x2, y2], ...])
    → (원본 크기 마스크 [물체 수, 후보 수, H, W], 모델이 스스로 매긴 IoU 점수 [물체 수, 후보 수], 이미지 Embedding)"""
    kw = {"input_points": [[[p] for p in points]], "input_labels": [[[1] for _ in points]]} if points is not None else {"input_boxes": [boxes]}
    inp = proc(images=image, return_tensors="pt", **kw).to(device)
    with torch.no_grad():
        if emb is None:
            emb = model.get_image_embeddings(inp["pixel_values"])           # 무거운 부분: 이미지당 한 번
        out = model(**{k: v for k, v in inp.items() if k not in ("pixel_values", "original_sizes", "reshaped_input_sizes")},
                    image_embeddings=emb, multimask_output=multimask)       # 가벼운 부분: Prompt마다
    args = [out.pred_masks.cpu(), inp["original_sizes"].cpu()]
    if "reshaped_input_sizes" in inp:                                       # SAM(원조)만 필요
        args.append(inp["reshaped_input_sizes"].cpu())
    return proc.post_process_masks(*args)[0], out.iou_scores[0].cpu(), emb


bus = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")

# 1. 점 하나, 박스 하나 — SAM ViT-B
mid, M, P = MODELS["SAM ViT-B"]
model, proc = M.from_pretrained(mid).eval().to(device), P.from_pretrained(mid)
print(f"SAM ViT-B 파라미터 {sum(p.numel() for p in model.parameters()):,}")
masks, scores, emb = segment(model, proc, bus, points=[[280, 600]], multimask=True)
print("점 (280, 600) 하나 → 마스크 후보 3개")
for k in range(3):
    print(f"  후보 {k}: 넓이 {int(masks[0, k].sum()):6,}화소, 모델이 매긴 IoU 점수 {float(scores[0, k]):.3f}")
bmask, bscore, _ = segment(model, proc, bus, emb=emb, boxes=[[33, 230, 806, 739]])
print(f"박스 (33, 230, 806, 739) → 마스크 1개: 넓이 {int(bmask[0, 0].sum()):,}화소, IoU 점수 {float(bscore[0, 0]):.3f}")
vis = np.array(bus)[:, :, ::-1].copy()
panels = []
for k, color in enumerate([(0, 0, 255), (0, 200, 0), (255, 0, 0)]):
    v = vis.copy()
    v[masks[0, k].numpy()] = (0.45 * v[masks[0, k].numpy()] + 0.55 * np.array(color)).astype(np.uint8)
    cv2.circle(v, (280, 600), 12, (0, 255, 255), -1)
    panels.append(v)
v = vis.copy()
v[bmask[0, 0].numpy()] = (0.45 * v[bmask[0, 0].numpy()] + 0.55 * np.array((255, 0, 255))).astype(np.uint8)
cv2.rectangle(v, (33, 230), (806, 739), (0, 255, 255), 4)
panels.append(v)

# 2. 시간 — Image Encoder(이미지당 한 번)와 Mask Decoder(Prompt마다)
print(f"\n{'모델':<18}{'파라미터':>12}{'Encoder(ms)':>13}{'Decoder 1회(ms)':>17}")
loaded = {}
for name, (mid, M, P) in MODELS.items():
    model, proc = M.from_pretrained(mid).eval().to(device), P.from_pretrained(mid)
    loaded[name] = (model, proc)
    inp = proc(images=bus, input_points=[[[[280, 600]]]], input_labels=[[[1]]], return_tensors="pt").to(device)
    rest = {k: v for k, v in inp.items() if k not in ("pixel_values", "original_sizes", "reshaped_input_sizes")}
    with torch.no_grad():
        for _ in range(3):
            emb = model.get_image_embeddings(inp["pixel_values"])
            model(**rest, image_embeddings=emb)
        sync()
        t = time.perf_counter()
        for _ in range(20):
            emb = model.get_image_embeddings(inp["pixel_values"])
        sync()
        enc = (time.perf_counter() - t) / 20 * 1000
        t = time.perf_counter()
        for _ in range(20):
            model(**rest, image_embeddings=emb)
        sync()
        dec = (time.perf_counter() - t) / 20 * 1000
    print(f"{name:<18}{sum(p.numel() for p in model.parameters()):>12,}{enc:>13.1f}{dec:>17.1f}")

# 3. COCO 500장 — 정답 박스 / 정답 마스크 안쪽 점 하나를 Prompt로
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
by_img = defaultdict(list)
for a in inst["annotations"]:
    if not a["iscrowd"]:
        by_img[a["image_id"]].append(a)


def size_of(area):
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")       # COCO의 크기 구분


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


print(f"\nCOCO val2017 {len(inst['images'])}장, 물체 {sum(len(v) for v in by_img.values())}개 (crowd 제외)")
print(f"{'모델':<18}{'Prompt':<10}{'평균 IoU':>9}{'small':>8}{'medium':>8}{'large':>8}{'시간(초)':>9}")
for name, (model, proc) in loaded.items():
    res = {"박스": defaultdict(list), "점 하나": defaultdict(list), "점 (최선)": defaultdict(list)}
    t0 = time.perf_counter()
    for im in inst["images"]:
        anns = by_img[im["id"]]
        if not anns:
            continue
        image = Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")
        gts = [mask_utils.decode(mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])).max(axis=2).astype(bool)
               if isinstance(a["segmentation"], list) else mask_utils.decode(a["segmentation"]).astype(bool) for a in anns]
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        points = []
        for g in gts:                                                          # 정답 마스크에서 가장자리로부터 가장 먼 점
            d = cv2.distanceTransform(np.pad(g, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
            y, x = np.unravel_index(d.argmax(), d.shape)
            points.append([int(x), int(y)])
        bm, _, emb = segment(model, proc, image, boxes=boxes)
        pm, ps, _ = segment(model, proc, image, emb=emb, points=points, multimask=True)
        for i, (a, g) in enumerate(zip(anns, gts)):
            res["박스"][size_of(a["area"])].append(iou(bm[i, 0].numpy(), g))
            res["점 하나"][size_of(a["area"])].append(iou(pm[i, int(ps[i].argmax())].numpy(), g))   # 점수가 가장 높은 후보
            res["점 (최선)"][size_of(a["area"])].append(max(iou(pm[i, k].numpy(), g) for k in range(3)))   # 후보 3개 중 정답에 가장 가까운 것
    sec = time.perf_counter() - t0
    for prompt, r in res.items():
        allv = [v for vs in r.values() for v in vs]
        print(f"{name:<18}{prompt:<10}{np.mean(allv):>9.3f}" + "".join(f"{np.mean(r[s]):>8.3f}" for s in ["small", "medium", "large"])
              + f"{sec:>9.0f}")
print("  (물체 수 small·medium·large: " + ", ".join(str(sum(size_of(a["area"]) == s for v in by_img.values() for a in v)) for s in ["small", "medium", "large"]) + ")")

# 4. 자동 마스크 생성 — 32x32 격자의 점마다 Prompt를 넣고, 점수가 낮거나 겹치는 마스크를 걸러 낸다
gen = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if device == "cuda" else -1)
gen(bus, points_per_batch=64)                                                   # 워밍업
t = time.perf_counter()
auto = gen(bus, points_per_batch=64)
print(f"\n자동 마스크 생성 (SAM ViT-B, 점 32x32=1,024개): 마스크 {len(auto['masks'])}개, {time.perf_counter() - t:.1f}초")
areas = sorted((int(m.sum()) for m in auto["masks"]), reverse=True)
print(f"  넓이 큰 순서 5개: {areas[:5]} | 1,000화소 미만 {sum(a < 1000 for a in areas)}개")
v = vis.copy()
rng = np.random.default_rng(0)
for m in sorted(auto["masks"], key=lambda m: -int(m.sum())):                      # 큰 것부터 칠해 작은 것이 위에 보이게
    m = np.asarray(m, dtype=bool)
    v[m] = (0.4 * v[m] + 0.6 * rng.integers(40, 255, 3)).astype(np.uint8)
panels.append(v)
small = [cv2.resize(p, (405, 540)) for p in panels]
cv2.imwrite(str(OUT / "08_2_prompts.jpg"), np.hstack(small))
print(f"→ {OUT / '08_2_prompts.jpg'} (점 후보 0·1·2 | 박스 | 자동)")
