"""10-4. 실습: Visual Grounding과 공간 이해

1) "사람과 버스의 위치를 찾아라" — 좌표(JSON) 출력과 시각화. Qwen3.5의 좌표는 0~1000으로 정규화된 값
2) 공간 질문 — 05-5 영상 1,449번째 프레임의 "빨간 자동차의 오른쪽에 있는 것"
3) COCO val2017 500장 — 07-4·08-4·09-2와 같은 질문 1,985개(있는 물체 찾기·없는 물체 찾기)를 좌표로 답하게 해 채점

실행 (저장소 루트에서, vLLM 서버를 띄운 뒤):
    python ch10_vlm/10_4_grounding_spatial.py
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, model_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch10"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--skip-coco", action="store_true", help="COCO 채점(시간이 오래 걸림)을 건너뛴다")
args = ap.parse_args()
B = args.base_url
model = model_name(B)
print(f"모델: {model}")


def parse_boxes(text, size):
    """답에서 {"label", "bbox_2d"} 목록을 꺼내 원본 화소 좌표로 바꾼다 (0~1000 → 화소)"""
    w, h = size
    boxes = []
    for obj in re.findall(r"\{[^{}]*\}", text):                                  # JSON 객체 { ... } 하나씩
        nums = re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', obj)
        label = re.search(r'"label"\s*:\s*"([^"]*)"', obj)
        v = [float(x) for x in re.findall(r"-?\d+\.?\d*", nums.group(1))] if nums else []
        if len(v) == 4:
            boxes.append((label.group(1) if label else "", [v[0] / 1000 * w, v[1] / 1000 * h, v[2] / 1000 * w, v[3] / 1000 * h]))
    return boxes


# 1. 위치 찾기와 시각화
bus = Image.open(ROOT / "data" / "images" / "bus.jpg")
r = ask('Locate every person and the bus. Output a JSON list like [{"label": "person", "bbox_2d": [x1, y1, x2, y2]}].',
        [bus], max_tokens=600, base_url=B)
boxes = parse_boxes(r["text"], bus.size)
print(f"[bus.jpg] {r['seconds']:.2f}초, 박스 {len(boxes)}개")
vis = cv2.cvtColor(np.array(bus.convert("RGB")), cv2.COLOR_RGB2BGR)
for label, b in boxes:
    print(f"  {label:<7} [{', '.join(f'{v:.0f}' for v in b)}]")
    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 4)
    cv2.putText(vis, label, (int(b[0]) + 5, int(b[1]) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
cv2.imwrite(str(OUT / f"10_4_bus_{model.split('/')[-1]}.jpg"), vis)

# 2. 공간 질문 — 05-5 영상 1,449번째 프레임
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
frame = Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
for prompt in ['Locate the red car. Output a JSON list like [{"label": "red car", "bbox_2d": [x1, y1, x2, y2]}].',
               "What object is directly to the right of the red car? Answer in one short sentence.",
               "Is the red car partly hidden by another vehicle? If so, which one? Answer in one sentence."]:
    r = ask(prompt, [frame], max_tokens=300, base_url=B, max_side=1280)       # 좌표는 0~1000 비율이라 줄여도 그대로 쓴다
    shown = r["text"] if "bbox_2d" not in r["text"] else " | ".join(f"{l} [{', '.join(f'{v:.0f}' for v in b)}]" for l, b in parse_boxes(r["text"], frame.size))
    print(f"[1449프레임] {prompt[:70]}\n  → {shown}  ({r['seconds']:.2f}초)")
if args.skip_coco:
    raise SystemExit


# 3. COCO 500장 — 07-4·08-4·09-2와 같은 질문, 같은 채점 (박스 IoU 0.5)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


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
rng = random.Random(0)                                                        # 앞 장들과 같은 "없는 물체" 질문
s = defaultdict(int)
t0 = time.perf_counter()
for im in inst["images"]:
    image = Image.open(COCO_DIR / "images" / im["file_name"])
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])
    for cls in present + [absent]:
        r = ask(f'Find every {cls} in this image. Output a JSON list like [{{"label": "{cls}", "bbox_2d": [x1, y1, x2, y2]}}]. '
                f"If there is no {cls}, output [].", [image], max_tokens=1024, base_url=B)
        boxes = [b for _, b in parse_boxes(r["text"], image.size)]
        s["출력 토큰"] += r["completion_tokens"]
        if cls == absent:
            s["없는 물체 질문"] += 1
            s["지어낸 횟수"] += int(len(boxes) > 0)
        else:
            s["정답 박스"] += len(gt[im["id"]][cls])
            s["예측 박스"] += len(boxes)
            s["맞힌 박스"] += match(boxes, gt[im["id"]][cls])
sec = time.perf_counter() - t0
print(f"\nCOCO val2017 500장, 질문 {s['없는 물체 질문'] + sum(len(gt[i['id']]) for i in inst['images'])}개, {sec:.0f}초 (질문당 {sec / 1985:.2f}초)")
print(f"  {model}: 재현율 {s['맞힌 박스'] / s['정답 박스']:.1%} | 정밀도 {s['맞힌 박스'] / max(1, s['예측 박스']):.1%} | "
      f"없는 물체를 찾아냄 {s['지어낸 횟수'] / s['없는 물체 질문']:.1%} (정답 {s['정답 박스']}, 예측 {s['예측 박스']}, 맞힘 {s['맞힌 박스']})")
