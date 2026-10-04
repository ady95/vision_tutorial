"""08-4. SAM 3: 개념 Prompt와 영상 Tracking

1) 글 Prompt — "person", "bus", "sunglasses", 그리고 사진에 없는 "giraffe"
   이미지 특징은 한 번만 뽑고 글만 바꿔 넣는다
2) Visual Exemplar — 박스 하나를 예시로 주면 "이것과 같은 것"을 모두 찾는다
3) COCO val2017 500장 — 07-4와 같은 방식(있는 물체 찾기·없는 물체 찾기)으로 채점
4) 영상 — "red car"를 글로 지정해 05-5 교통 영상에서 계속 추적하기 (1,380~1,559번째 프레임)

facebook/sam3 가중치는 Hugging Face에서 접근 승인을 받아야 내려받을 수 있다.
    1) https://huggingface.co/facebook/sam3 에서 접근 요청 → 승인
    2) 실행할 컴퓨터에서 hf auth login (토큰 입력)

실행 (저장소 루트에서, data/download_coco_val.py와 data/download_samples.py 실행 후):
    python ch08_sam/08_4_sam3.py
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
from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch08"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").eval().to(device)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def find(image_inputs, vision, text=None, boxes=None, threshold=0.5):
    """이미지 특징(vision)은 그대로 두고 글 또는 예시 박스만 바꿔 넣는다 → 점수·박스·마스크"""
    t = proc(text=text if text is not None else "visual", return_tensors="pt").to(device)   # 박스만 줄 때는 "visual"이 기본 글
    kw = {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}
    if boxes is not None:
        b = proc(images=None, input_boxes=[boxes], input_boxes_labels=[[1] * len(boxes)],
                 original_sizes=image_inputs["original_sizes"], return_tensors="pt").to(device)
        kw.update(input_boxes=b["input_boxes"], input_boxes_labels=b["input_boxes_labels"])
    with torch.no_grad():
        out = model(vision_embeds=vision, **kw)
    return proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                   target_sizes=image_inputs["original_sizes"].tolist())[0]


def encode(image):
    inp = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return inp, model.get_vision_features(pixel_values=inp["pixel_values"])


# 1. 글 Prompt
print(f"SAM 3 파라미터 {sum(p.numel() for p in model.parameters()):,}")
bus = Image.open(ROOT / "data" / "images" / "bus.jpg").convert("RGB")
encode(bus)                                                                  # 워밍업
sync()
t = time.perf_counter()
inp, vision = encode(bus)
sync()
print(f"이미지 특징 뽑기 {(time.perf_counter() - t) * 1000:.0f} ms (이미지당 한 번)")
for text in ["person", "bus", "sunglasses", "giraffe"]:
    t = time.perf_counter()
    r = find(inp, vision, text=text)
    sync()
    print(f"  '{text}': {len(r['scores'])}개 {(time.perf_counter() - t) * 1000:4.0f} ms | "
          + " ".join(f"{float(s):.2f}[{', '.join(f'{v:.0f}' for v in b)}]" for s, b in zip(r["scores"], r["boxes"])))

# 2. Visual Exemplar — 05-5의 1,449번째 프레임에서 승합차 박스 하나를 예시로
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1448)
ok, frame = cap.read()
frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
finp, fvis = encode(frame_img)
example = [1027, 587, 1467, 1045]                                            # 오른쪽 차로의 흰 승합차
for name, kw in [("예시 박스만", {"boxes": [example]}), ("글 'white van'", {"text": "white van"}), ("글 'car'", {"text": "car"})]:
    r = find(finp, fvis, **kw)
    print(f"[1449프레임] {name:<16}: {len(r['scores'])}개 | " + " ".join(f"{float(s):.2f}[{', '.join(f'{v:.0f}' for v in b)}]" for s, b in list(zip(r["scores"], r["boxes"]))[:6]))


# 3. COCO 500장 — 있는 물체 찾기, 없는 물체 찾기 (07-4와 같은 질문, 같은 채점)
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
rng = random.Random(0)                                                        # 07-4와 같은 seed → 같은 "없는 물체" 질문
s = defaultdict(int)
t0 = time.perf_counter()
for im in inst["images"]:
    image = Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB")
    present = list(gt[im["id"]])
    absent = rng.choice([n for n in names.values() if n not in present])
    cinp, cvis = encode(image)
    for cls in present + [absent]:
        boxes = find(cinp, cvis, text=cls)["boxes"].tolist()
        if cls == absent:
            s["없는 물체 질문"] += 1
            s["지어낸 횟수"] += int(len(boxes) > 0)
        else:
            s["정답 박스"] += len(gt[im["id"]][cls])
            s["예측 박스"] += len(boxes)
            s["맞힌 박스"] += match(boxes, gt[im["id"]][cls])
print(f"\nCOCO val2017 {len(inst['images'])}장 (신뢰도 0.5 이상), {time.perf_counter() - t0:.0f}초")
print(f"  SAM 3: 재현율 {s['맞힌 박스'] / s['정답 박스']:.1%} | 정밀도 {s['맞힌 박스'] / s['예측 박스']:.1%} | "
      f"없는 물체를 찾아냄 {s['지어낸 횟수'] / s['없는 물체 질문']:.1%} (정답 {s['정답 박스']}, 예측 {s['예측 박스']}, 맞힘 {s['맞힌 박스']})")

# 4. 영상 — "red car"를 계속 추적 (bfloat16으로 메모리·시간 절약)
del model
torch.cuda.empty_cache()
vproc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
vmodel = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16).eval()
START, N = 1379, 180                                                          # 1,380~1,559번째 프레임 (0부터 센다)
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
frames = [Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)) for _ in range(N)]
session = vproc.init_video_session(video=frames, inference_device=device, processing_device="cpu",
                                   video_storage_device="cpu", dtype=torch.bfloat16)
session = vproc.add_text_prompt(inference_session=session, text="red car")
tracks = defaultdict(list)                                                    # 물체 ID → [(프레임, 박스)]
masks_at = {}
t0 = time.perf_counter()
for o in vmodel.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=N):
    p = vproc.postprocess_outputs(session, o)
    for oid, box in zip(p["object_ids"].tolist(), p["boxes"].tolist()):
        tracks[oid].append((o.frame_idx, box))
    if o.frame_idx in (20, 70, 120):
        masks_at[o.frame_idx] = (p["object_ids"].tolist(), p["masks"].cpu().numpy())
sync()
sec = time.perf_counter() - t0
print(f"\n영상 'red car' 추적: {N}프레임 {sec:.0f}초 (프레임당 {sec / N:.2f}초), 물체 ID {len(tracks)}개")
for oid, tr in sorted(tracks.items()):
    (f0, b0), (f1, b1) = tr[0], tr[-1]
    print(f"  ID {oid}: 프레임 {START + 1 + f0}~{START + 1 + f1} 중 {len(tr)}프레임 | 처음 박스 [{', '.join(f'{v:.0f}' for v in b0)}] → 마지막 [{', '.join(f'{v:.0f}' for v in b1)}]")

# 그림 — 세 시점에서 추적 중인 마스크 (번호판이 있을 자리는 모자이크)
yolo = YOLO("yolo26s.pt")
tiles = []
for k, (ids, ms) in sorted(masks_at.items()):
    v = cv2.cvtColor(np.array(frames[k]), cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2 in yolo(v, classes=[2, 5, 7], conf=0.25, verbose=False)[0].boxes.xyxy.int().tolist():
        w, h = x2 - x1, y2 - y1
        px1, px2, py1, py2 = x1 + w // 4, x2 - w // 4, y1 + h * 62 // 100, y1 + h * 95 // 100
        patch = v[py1:py2, px1:px2]
        if patch.size:
            v[py1:py2, px1:px2] = cv2.resize(cv2.resize(patch, (max(1, (px2 - px1) // 10), max(1, (py2 - py1) // 10))),
                                             (px2 - px1, py2 - py1), interpolation=cv2.INTER_NEAREST)
    for oid, m in zip(ids, ms):
        m = m.astype(bool)
        v[m] = (0.35 * v[m] + 0.65 * np.array((255, 255, 0))).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(v, f"ID {oid}", (int(xs.min()), int(ys.min()) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    crop = v[220:900, 380:1500].copy()
    cv2.putText(crop, f"frame {START + 1 + k}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    tiles.append(cv2.resize(crop, (560, 340)))
cv2.imwrite(str(OUT / "08_4_red_car_tracking.jpg"), np.hstack(tiles))
print(f"→ {OUT / '08_4_red_car_tracking.jpg'}")
