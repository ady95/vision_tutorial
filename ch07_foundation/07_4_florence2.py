"""07-4. Florence-2: 하나의 모델, 여러 Task — 프롬프트만 바꿔 캡션·검출·Grounding

1) 같은 사진에 Task 프롬프트 여섯 가지 — Florence-2 base와 large
2) 05-5의 승합차 프레임 — 검출과 문장으로 찾기
3) COCO val2017 500장 — 사진에 있는 물체 이름으로 찾기(재현율·정밀도)와 없는 물체 이름으로 찾기(지어내기)
   YOLO26s의 같은 클래스 결과와 같은 기준(IoU 0.5)으로 비교

실행 (저장소 루트에서, data/download_coco_val.py와 data/download_samples.py 실행 후):
    python ch07_foundation/07_4_florence2.py
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
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def load(name):
    mid = f"florence-community/Florence-2-{name}"
    return AutoProcessor.from_pretrained(mid), Florence2ForConditionalGeneration.from_pretrained(mid, dtype=dtype).eval().to(device)


def ask(proc, model, image, task, text=""):
    """Task 토큰(+글)을 프롬프트로 넣고, 생성된 토큰을 Task에 맞는 결과(글, 박스)로 바꾼다"""
    inp = proc(text=task + text, images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=256, num_beams=3, do_sample=False)
    return proc.post_process_generation(proc.batch_decode(ids, skip_special_tokens=False)[0], task=task, image_size=image.size)[task]


# 1. 같은 사진, 여섯 가지 Task
img = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")
TASKS = [("<CAPTION>", ""), ("<DETAILED_CAPTION>", ""), ("<OD>", ""),
         ("<CAPTION_TO_PHRASE_GROUNDING>", "a man wearing sunglasses"),
         ("<OPEN_VOCABULARY_DETECTION>", "person"), ("<OPEN_VOCABULARY_DETECTION>", "giraffe")]
for name in ["base", "large"]:
    proc, model = load(name)
    print(f"\nFlorence-2-{name}: 파라미터 {sum(p.numel() for p in model.parameters()):,}")
    ask(proc, model, img, "<CAPTION>")                           # 워밍업
    for task, text in TASKS:
        t = time.perf_counter()
        res = ask(proc, model, img, task, text)
        ms = (time.perf_counter() - t) * 1000
        if isinstance(res, str):
            shown = res
        else:
            labels = res.get("labels") or res.get("bboxes_labels")
            shown = " | ".join(f"{l} [{', '.join(f'{v:.0f}' for v in b)}]" for l, b in zip(labels, res["bboxes"]))
        print(f"  {task + (' ' + text if text else ''):<52} {ms:6.0f} ms  {shown}")

# 2. 05-5의 승합차 — Florence-2-large로
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1380)
ok, frame = cap.read()
van = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
od = ask(proc, model, van, "<OD>")
near = [(l, b) for l, b in zip(od["labels"], od["bboxes"]) if 1000 < (b[0] + b[2]) / 2 < 1425 and 500 < (b[1] + b[3]) / 2 < 760]
print("\n05-5 승합차 프레임 <OD>: " + " | ".join(f"{l} [{', '.join(f'{v:.0f}' for v in b)}]" for l, b in near))
g = ask(proc, model, van, "<CAPTION_TO_PHRASE_GROUNDING>", "a white van")
print("  'a white van' 찾기: " + " | ".join(f"{l} [{', '.join(f'{v:.0f}' for v in b)}]" for l, b in zip(g["labels"], g["bboxes"])))


# 3. COCO 500장 — 있는 물체 찾기, 없는 물체 찾기
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, gts):
    """IoU 0.5 이상인 짝을 하나씩 (겹친 정도가 큰 순서로)"""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g = set(), set()
    for v, i, j in pairs:
        if v >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j)
    return len(used_g)


inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
names = {c["id"]: c["name"] for c in inst["categories"]}
gt = defaultdict(lambda: defaultdict(list))                     # 이미지 → 클래스 이름 → 정답 박스들
for a in inst["annotations"]:
    if not a["iscrowd"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][names[a["category_id"]]].append([x, y, x + w, y + h])
yolo = YOLO("yolo26s.pt")
yolo_id = {v: k for k, v in yolo.names.items()}                 # YOLO 클래스 이름 → 번호 (COCO와 같은 이름)
rng = random.Random(0)
stats = {k: defaultdict(int) for k in ["Florence-2-large", "YOLO26s"]}
t0 = time.perf_counter()
for im in inst["images"]:
    image = Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])     # 이 사진에 없는 클래스 하나
    yr = yolo(image, conf=0.25, verbose=False)[0]
    for cls in present + [absent]:
        f_boxes = ask(proc, model, image, "<CAPTION_TO_PHRASE_GROUNDING>", cls)["bboxes"]
        y_boxes = [b for b, c in zip(yr.boxes.xyxy.tolist(), yr.boxes.cls.tolist()) if int(c) == yolo_id[cls]]
        for key, boxes in [("Florence-2-large", f_boxes), ("YOLO26s", y_boxes)]:
            s = stats[key]
            if cls == absent:
                s["없는 물체 질문"] += 1
                s["지어낸 횟수"] += int(len(boxes) > 0)
            else:
                s["정답 박스"] += len(gt[im["id"]][cls])
                s["예측 박스"] += len(boxes)
                s["맞힌 박스"] += match(boxes, gt[im["id"]][cls])
print(f"\nCOCO val2017 {len(inst['images'])}장, 있는 물체 질문 {sum(len(gt[i['id']]) for i in inst['images'])}개 + 없는 물체 질문 500개, {time.perf_counter() - t0:.0f}초")
print(f"{'':<18}{'재현율':>8}{'정밀도':>8}{'없는 물체를 찾아냄':>14}")
for key, s in stats.items():
    print(f"{key:<18}{s['맞힌 박스'] / s['정답 박스']:>8.1%}{s['맞힌 박스'] / max(1, s['예측 박스']):>8.1%}"
          f"{s['지어낸 횟수'] / s['없는 물체 질문']:>14.1%}  (정답 {s['정답 박스']}, 예측 {s['예측 박스']}, 맞힘 {s['맞힌 박스']})")
