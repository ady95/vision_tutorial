"""09-3. 실습: 자연어 기반 객체 분할 — Text → Grounding DINO → Box → SAM 2.1 → Mask

1) "빨간 자동차만 분할하라" — 05-5 영상의 1,449번째 프레임에서 두 방법 비교
   (가) Grounding DINO base + SAM 2.1 B+   (나) SAM 3 (08-4, 모델 하나로 글 → 마스크)
2) COCO val2017 500장 — 같은 질문(있는 물체·없는 물체)을 마스크 단위로 채점 (정답 마스크와 IoU 0.5 이상이면 맞음)

실행 (저장소 루트에서, data/download_coco_val.py·data/download_samples.py 실행 후, SAM 3는 08-4의 접근 승인 필요):
    python ch09_open_vocab/09_3_text_to_mask.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor, Sam3Model,
                          Sam3Processor)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch09"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval().to(device)
sam2_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus")
sam2 = Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus").eval().to(device)
sam3_proc = Sam3Processor.from_pretrained("facebook/sam3")
sam3 = Sam3Model.from_pretrained("facebook/sam3").eval().to(device)


def text_to_mask_gdino_sam(image, text, sam_emb=None):
    """(가) 글 → Grounding DINO 박스 → SAM 2.1 마스크. SAM 이미지 Embedding은 재사용할 수 있게 돌려준다"""
    inp = gd_proc(images=image, text=f"{text}.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.35, text_threshold=0.25,
                                                       target_sizes=[image.size[::-1]])[0]
    boxes = r["boxes"].tolist()
    if not boxes:
        return [], [], sam_emb
    s_inp = sam2_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        if sam_emb is None:
            sam_emb = sam2.get_image_embeddings(s_inp["pixel_values"])
        o = sam2(input_boxes=s_inp["input_boxes"], image_embeddings=sam_emb, multimask_output=False)
    masks = sam2_proc.post_process_masks(o.pred_masks.cpu(), s_inp["original_sizes"].cpu())[0][:, 0].numpy()
    return list(masks), r["scores"].tolist(), sam_emb


def text_to_mask_sam3(image, text, vision=None):
    """(나) SAM 3 — 글 → 마스크 (이미지 특징은 재사용)"""
    inp = sam3_proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        if vision is None:
            vision = sam3.get_vision_features(pixel_values=inp["pixel_values"])
        t = sam3_proc(text=text, return_tensors="pt").to(device)
        out = sam3(vision_embeds=vision, input_ids=t["input_ids"], attention_mask=t["attention_mask"])
    r = sam3_proc.post_process_instance_segmentation(out, threshold=0.5, mask_threshold=0.5,
                                                     target_sizes=inp["original_sizes"].tolist())[0]
    return list(r["masks"].cpu().numpy().astype(bool)), r["scores"].tolist(), vision


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


# 1. "빨간 자동차만 분할하라"
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
bgr = cap.read()[1]
frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
for fn in [text_to_mask_gdino_sam, text_to_mask_sam3]:
    fn(frame, "red car")                                                  # 워밍업
panels = []
for name, fn, color in [("G-DINO + SAM 2.1", text_to_mask_gdino_sam, (0, 255, 255)), ("SAM 3", text_to_mask_sam3, (255, 255, 0))]:
    sync()
    t = time.perf_counter()
    masks, scores, _ = fn(frame, "red car")
    sync()
    ms = (time.perf_counter() - t) * 1000
    desc = []
    v = bgr.copy()
    for m, s in zip(masks, scores):
        ys, xs = np.nonzero(m)
        desc.append(f"{s:.2f} 넓이 {int(m.sum()):,} 중심 ({xs.mean():.0f}, {ys.mean():.0f})")
        v[m] = (0.35 * v[m] + 0.65 * np.array(color)).astype(np.uint8)
    print(f"[1449프레임 'red car'] {name:<16} {ms:5.0f} ms, 마스크 {len(masks)}개: " + " | ".join(desc))
    for x1, y1, x2, y2 in YOLO("yolo26s.pt")(bgr, classes=[2, 5, 7], conf=0.25, verbose=False)[0].boxes.xyxy.int().tolist():
        w, h = x2 - x1, y2 - y1                                          # 번호판이 있을 자리는 모자이크
        px1, px2, py1, py2 = x1 + w // 4, x2 - w // 4, y1 + h * 62 // 100, y1 + h * 95 // 100
        patch = v[py1:py2, px1:px2]
        if patch.size:
            v[py1:py2, px1:px2] = cv2.resize(cv2.resize(patch, (max(1, (px2 - px1) // 10), max(1, (py2 - py1) // 10))),
                                             (px2 - px1, py2 - py1), interpolation=cv2.INTER_NEAREST)
    crop = v[220:900, 380:1500].copy()
    cv2.putText(crop, name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    panels.append(cv2.resize(crop, (560, 340)))
cv2.imwrite(str(OUT / "09_3_red_car.jpg"), np.hstack(panels))


# 2. COCO 500장 — 마스크 단위 채점
def mask_match(preds, gts):
    pairs = sorted((((p & g).sum() / max(1, (p | g).sum()), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g = set(), set()
    for v, i, j in pairs:
        if v >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_g)


inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
gt = defaultdict(lambda: defaultdict(list))
for a in inst["annotations"]:
    if not a["iscrowd"]:
        gt[a["image_id"]][names[a["category_id"]]].append(a)
rng = random.Random(0)                                                     # 07-4·08-4·09-2와 같은 "없는 물체" 질문
stats = {k: defaultdict(int) for k in ["G-DINO + SAM 2.1", "SAM 3"]}
times = defaultdict(float)
for im in inst["images"]:
    image = Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])
    cache = {"G-DINO + SAM 2.1": None, "SAM 3": None}
    for cls in present + [absent]:
        gts = [mask_utils.decode(mask_utils.frPyObjects(a["segmentation"], im["height"], im["width"])).max(axis=2).astype(bool)
               for a in gt[im["id"]][cls]] if cls != absent else []
        for name, fn in [("G-DINO + SAM 2.1", text_to_mask_gdino_sam), ("SAM 3", text_to_mask_sam3)]:
            t = time.perf_counter()
            masks, _, cache[name] = fn(image, cls, cache[name])
            sync()
            times[name] += time.perf_counter() - t
            s = stats[name]
            if cls == absent:
                s["없는 물체 질문"] += 1
                s["지어낸 횟수"] += int(len(masks) > 0)
            else:
                s["정답 마스크"] += len(gts)
                s["예측 마스크"] += len(masks)
                s["맞힌 마스크"] += mask_match(masks, gts)
print(f"\nCOCO val2017 {len(inst['images'])}장, 마스크 단위 채점 (정답 마스크와 IoU 0.5 이상)")
print(f"{'':<20}{'재현율':>8}{'정밀도':>8}{'없는 물체를 찾아냄':>14}{'시간(초)':>9}")
for name, s in stats.items():
    print(f"{name:<20}{s['맞힌 마스크'] / s['정답 마스크']:>8.1%}{s['맞힌 마스크'] / max(1, s['예측 마스크']):>8.1%}"
          f"{s['지어낸 횟수'] / s['없는 물체 질문']:>14.1%}{times[name]:>9.0f}  (정답 {s['정답 마스크']}, 예측 {s['예측 마스크']}, 맞힘 {s['맞힌 마스크']})")
