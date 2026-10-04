"""09-2. 실습: Grounding DINO로 텍스트 객체 찾기

1) 글로 찾기 — 물체 이름, 꾸밈말이 붙은 표현, 사진에 없는 것
2) 05-5의 1,449번째 프레임 — "white van", "red car", "black SUV", "police car"
3) COCO val2017 500장 — 07-4·08-4와 같은 질문(있는 물체 찾기·없는 물체 찾기)
   Grounding DINO tiny·base는 Box Threshold 0.25·0.35·0.5로, YOLOE-26s(Open-Vocabulary YOLO)와 함께 채점

실행 (저장소 루트에서, data/download_coco_val.py와 data/download_samples.py 실행 후):
    python ch09_open_vocab/09_2_grounding_dino.py
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"G-DINO tiny": "IDEA-Research/grounding-dino-tiny", "G-DINO base": "IDEA-Research/grounding-dino-base"}
loaded = {name: (AutoProcessor.from_pretrained(mid), AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval().to(device))
          for name, mid in MODELS.items()}


def ground(name, image, text, box_threshold=0.35, text_threshold=0.25):
    """text는 소문자 표현을 마침표로 구분 ("person. bus.") → [(표현, 점수, 박스)]"""
    proc, model = loaded[name]
    inp = proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_threshold, text_threshold=text_threshold,
                                                    target_sizes=[image.size[::-1]])[0]
    return [(l, float(s), b.tolist()) for s, l, b in zip(r["scores"], r["text_labels"], r["boxes"])]


def show(items):
    return " | ".join(f"{l} {s:.2f} [{', '.join(f'{v:.0f}' for v in b)}]" for l, s, b in items) or "(없음)"


# 1. 글로 찾기
bus = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")
for name, (proc, model) in loaded.items():
    print(f"{name}: 파라미터 {sum(p.numel() for p in model.parameters()):,}")
    ground(name, bus, "person.")                                         # 워밍업
    for text in ["person. bus.", "a man wearing sunglasses.", "sunglasses.", "giraffe."]:
        t = time.perf_counter()
        items = ground(name, bus, text)
        print(f"  '{text}' {(time.perf_counter() - t) * 1000:.0f} ms: {show(items)}")

# 2. 05-5의 1,449번째 프레임
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
frame = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
print("\n05-5 영상 1,449번째 프레임 (G-DINO base)")
for text in ["white van.", "red car.", "black suv.", "police car."]:
    items = ground("G-DINO base", frame, text)
    print(f"  '{text}' {len(items)}개: {show(items[:6])}")


# 3. COCO 500장 — 07-4·08-4와 같은 질문, 같은 채점
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, gts):
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
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
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][names[a["category_id"]]].append([x, y, x + w, y + h])
yoloe = YOLOE("yoloe-26s-seg.pt")
yoloe.set_classes(list(names.values()))                                  # 80종의 이름을 글로 넣어 클래스를 정한다
THRESHOLDS = [0.25, 0.35, 0.5]
stats = defaultdict(lambda: defaultdict(int))
times = defaultdict(float)
rng = random.Random(0)                                                    # 07-4·08-4와 같은 "없는 물체" 질문
for im in inst["images"]:
    image = Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])
    t = time.perf_counter()
    yr = yoloe(image, conf=0.25, verbose=False)[0]
    times["YOLOE-26s"] += time.perf_counter() - t
    for cls in present + [absent]:
        preds = {"YOLOE-26s": [b for b, c in zip(yr.boxes.xyxy.tolist(), yr.boxes.cls.tolist()) if yr.names[int(c)] == cls]}
        for name in MODELS:
            t = time.perf_counter()
            items = ground(name, image, f"{cls}.", box_threshold=min(THRESHOLDS))       # 가장 낮은 기준으로 한 번 → 기준별로 거른다
            times[name] += time.perf_counter() - t
            for th in THRESHOLDS:
                preds[f"{name} @{th}"] = [b for l, s, b in items if s >= th]
        for key, boxes in preds.items():
            s = stats[key]
            if cls == absent:
                s["없는 물체 질문"] += 1
                s["지어낸 횟수"] += int(len(boxes) > 0)
            else:
                s["정답 박스"] += len(gt[im["id"]][cls])
                s["예측 박스"] += len(boxes)
                s["맞힌 박스"] += match(boxes, gt[im["id"]][cls])
print(f"\nCOCO val2017 {len(inst['images'])}장, 질문 {stats['YOLOE-26s']['없는 물체 질문'] + sum(len(gt[i['id']]) for i in inst['images'])}개")
print(f"{'':<22}{'재현율':>8}{'정밀도':>8}{'없는 물체를 찾아냄':>14}")
for key, s in stats.items():
    print(f"{key:<22}{s['맞힌 박스'] / s['정답 박스']:>8.1%}{s['맞힌 박스'] / max(1, s['예측 박스']):>8.1%}{s['지어낸 횟수'] / s['없는 물체 질문']:>14.1%}")
print("총 시간: " + " | ".join(f"{k} {v:.0f}초" for k, v in times.items()) + " (YOLOE는 80종을 한 번에, Grounding DINO는 질문마다 한 번)")
