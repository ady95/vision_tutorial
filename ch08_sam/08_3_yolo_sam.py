"""08-3. 실습: Detection + SAM — YOLO의 박스를 SAM에 넘겨 정밀한 마스크 얻기

1) COCO val2017 500장: YOLO26s-seg가 찾은 물체마다
   (가) YOLO26s-seg 자신의 마스크와 (나) 그 박스를 SAM 2.1에 넣어 얻은 마스크를
   정답 마스크와 비교한다 (같은 물체, 같은 박스에서 출발하므로 마스크 품질만 비교된다)
2) 한 장 처리 시간 비교
3) 그림 — 한 사진에서 두 마스크의 외곽선 비교

실행 (저장소 루트에서, data/download_coco_val.py 실행 후):
    python ch08_sam/08_3_yolo_sam.py
    python ch08_sam/08_3_yolo_sam.py --image 내_차량_사진.jpg
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO
from ultralytics.data.converter import coco80_to_coco91_class

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch08"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(ROOT / "data" / "images" / "bus.jpg"), help="외곽선 비교 그림에 쓸 사진")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
TO_COCO = coco80_to_coco91_class()

yolo = YOLO("yolo26s-seg.pt")
sams = {name: (Sam2Model.from_pretrained(mid).eval().to(device), Sam2Processor.from_pretrained(mid))
        for name, mid in [("SAM 2.1 B+", "facebook/sam2.1-hiera-base-plus"), ("SAM 2.1 T", "facebook/sam2.1-hiera-tiny")]}


def yolo_masks(r, shape):
    """YOLO 마스크를 원본 해상도로 — masks.xy 다각형을 칠한다 (05-2와 같은 방법)"""
    out = []
    for poly in r.masks.xy:
        m = np.zeros(shape, np.uint8)
        if len(poly):
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        out.append(m.astype(bool))
    return out


def sam_masks(model, proc, image, boxes):
    inp = proc(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**{k: v for k, v in inp.items() if k != "original_sizes"}, multimask_output=False)
    return [m[0].numpy() for m in proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def mask_iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


# 1. COCO 500장 — 같은 박스에서 출발한 두 마스크를 정답과 비교
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
by_img = defaultdict(list)
for a in inst["annotations"]:
    if not a["iscrowd"]:
        by_img[a["image_id"]].append(a)
scores = defaultdict(lambda: defaultdict(list))                 # 방법 → 크기 → IoU 목록
times = defaultdict(float)
n_det = 0
for im in inst["images"]:
    path = COCO_DIR / "images" / im["file_name"]
    image = Image.open(path).convert("RGB")
    t = time.perf_counter()
    r = yolo(path, conf=0.25, device=device, verbose=False)[0]
    times["YOLO26s-seg"] += time.perf_counter() - t
    if r.masks is None:
        continue
    boxes, classes = r.boxes.xyxy.tolist(), [TO_COCO[int(c)] for c in r.boxes.cls]
    preds = {"YOLO26s-seg 마스크": yolo_masks(r, (im["height"], im["width"]))}
    for name, (model, proc) in sams.items():
        t = time.perf_counter()
        preds[f"YOLO 박스 → {name}"] = sam_masks(model, proc, image, boxes)
        times[name] += time.perf_counter() - t
    used = set()
    for i, (b, c) in enumerate(zip(boxes, classes)):           # 검출마다 같은 클래스의 정답 하나와 짝 (박스 IoU 0.5 이상, 가장 큰 것)
        cand = [(box_iou(b, [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]), j)
                for j, a in enumerate(by_img[im["id"]]) if a["category_id"] == c and j not in used]
        best = max(cand, default=(0, -1))
        if best[0] < 0.5:
            continue
        used.add(best[1])
        a = by_img[im["id"]][best[1]]
        rle = mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])
        gt = mask_utils.decode(rle).max(axis=2).astype(bool) if isinstance(a["segmentation"], list) else mask_utils.decode(a["segmentation"]).astype(bool)
        size = "small" if a["area"] < 32 ** 2 else ("medium" if a["area"] < 96 ** 2 else "large")
        for name, ms in preds.items():
            scores[name][size].append(mask_iou(ms[i], gt))
        n_det += 1

print(f"COCO val2017 {len(inst['images'])}장, 정답과 짝지은 검출 {n_det}개 (YOLO26s-seg, 신뢰도 0.25 이상, 같은 클래스·박스 IoU 0.5 이상)")
print(f"{'마스크':<26}{'평균 IoU':>9}{'small':>8}{'medium':>8}{'large':>8}")
for name, s in scores.items():
    allv = [v for vs in s.values() for v in vs]
    print(f"{name:<26}{np.mean(allv):>9.3f}" + "".join(f"{np.mean(s[k]):>8.3f}" for k in ["small", "medium", "large"]))
yolo_all = np.array([v for k in ["small", "medium", "large"] for v in scores["YOLO26s-seg 마스크"][k]])
for name in sams:
    sam_all = np.array([v for k in ["small", "medium", "large"] for v in scores[f"YOLO 박스 → {name}"][k]])
    print(f"  {name}가 YOLO 마스크보다 IoU가 높은 물체: {(sam_all > yolo_all).mean():.1%}, 0.1 이상 높은 물체 {(sam_all - yolo_all > 0.1).mean():.1%}, "
          f"0.1 이상 낮은 물체 {(yolo_all - sam_all > 0.1).mean():.1%}")

# 2. 시간 — 한 장 평균 (파일 읽기·전처리·후처리 포함)
n = len(inst["images"])
print(f"\n한 장 평균: YOLO26s-seg {times['YOLO26s-seg'] / n * 1000:.1f} ms"
      + "".join(f" | + {name} {times[name] / n * 1000:.1f} ms" for name in sams))

# 3. 그림 — 한 사진에서 YOLO 마스크(노랑)와 SAM 2.1 B+ 마스크(빨강)의 외곽선
img = cv2.imread(args.image)
r = yolo(img, conf=0.25, classes=[0, 2, 5, 7], device=device, verbose=False)[0]
ym = yolo_masks(r, img.shape[:2])
sm = sam_masks(*sams["SAM 2.1 B+"], Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), r.boxes.xyxy.tolist())
vis = img.copy()
for a, b in zip(ym, sm):
    for m, color in [(a, (0, 255, 255)), (b, (0, 0, 255))]:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, color, max(2, img.shape[1] // 400))
    print(f"[{Path(args.image).name}] 물체 마스크 넓이 YOLO {int(a.sum()):,} / SAM {int(b.sum()):,} 화소, 서로의 IoU {mask_iou(a, b):.3f}")
cv2.imwrite(str(OUT / f"08_3_{Path(args.image).stem}_outline.jpg"), vis)
