"""00-3. 빠른 시작: 자동차 사진 한 장, 여섯 가지 방법

05-5 교통 영상의 1,449번째 프레임 한 장을 이 책의 여섯 가지 방법으로 분석하고, 출력의 형태(화소·박스·마스크·벡터·문장)와 시간을 비교한다.
1) OpenCV (02장)        — 색으로 빨간 영역 찾기 → 화소
2) YOLO26s (04장)       — 차량 검출 → 박스
3) SAM 2.1 (08장)       — YOLO 박스 → 정밀한 마스크
4) Grounding DINO (09장) — 글 "black SUV" → 박스
5) CLIP (07장)          — 사진 → 벡터, COCO 500장에서 비슷한 사진 찾기
6) Qwen3.5 (10장)       — 사진 → 문장

실행 (저장소 루트에서, data/download_samples.py·data/download_coco_val.py 실행 후):
    uv pip install -e ".[dl,foundation,vlm]"
    python ch00_setup/00_3_quickstart.py
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                          Sam2Processor)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
OUT = ROOT / "outputs" / "ch00"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=1449)
ap.add_argument("--text", default="black SUV", help="Grounding DINO에 줄 글")
ap.add_argument("--vlm", default="Qwen/Qwen3.5-2B")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


def timed(fn, *a):
    """워밍업 한 번 뒤 다시 재서 (결과, ms)를 돌려준다"""
    fn(*a)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn(*a)
    if device == "cuda":
        torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1000


cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
bgr = cap.read()[1]
image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
H, W = bgr.shape[:2]
print(f"사진: 05-5 교통 영상 {args.frame}번째 프레임, {W}x{H}")
summary = []


# 1. OpenCV — 빨간색 화소를 찾아 덩어리로 묶는다 (02장)
def red_regions(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = sorted((s for s in stats[1:] if s[4] >= 300), key=lambda s: -s[4])    # 300화소 이상인 덩어리
    return mask, blobs


(red_mask, blobs), ms = timed(red_regions, bgr)
print(f"\n[1 OpenCV] 빨간 화소 {int((red_mask > 0).sum()):,}개 (사진의 {(red_mask > 0).mean():.2%}), "
      f"300화소 이상 덩어리 {len(blobs)}개 | {ms:.1f} ms")
for x, y, w, h, area in blobs[:4]:
    print(f"  덩어리 ({x}, {y}) {w}x{h}, {area:,}화소")
summary.append(("OpenCV", "화소 (빨간색인가)", f"덩어리 {len(blobs)}개", ms))

# 2. YOLO26s — 차량 박스 (04장)
yolo = YOLO("yolo26s.pt")


def detect(bgr):
    return yolo(bgr, classes=[2, 5, 7], conf=0.4, verbose=False)[0]                # 2 car, 5 bus, 7 truck


r, ms = timed(detect, bgr)
boxes = r.boxes.xyxy.int().tolist()
names = [r.names[c] for c in r.boxes.cls.int().tolist()]
print(f"\n[2 YOLO26s] 차량 {len(boxes)}대: " + ", ".join(f"{n} {names.count(n)}" for n in sorted(set(names)))
      + f" | {ms:.1f} ms")
for b, n, c in list(zip(boxes, names, r.boxes.conf.tolist()))[:3]:
    print(f"  {n} {c:.2f} {b}")
summary.append(("YOLO26s", "박스 + 클래스", f"차량 {len(boxes)}대", ms))

# 3. SAM 2.1 — YOLO 박스마다 정밀한 마스크 (08장)
sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus").eval().to(device)


def box_to_mask(image, boxes):
    inp = sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        o = sam(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"], multimask_output=False)
    return sam_proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0][:, 0].numpy()


masks, ms = timed(box_to_mask, image, boxes)
fill = [m.sum() / ((x2 - x1) * (y2 - y1)) for m, (x1, y1, x2, y2) in zip(masks, boxes)]
print(f"\n[3 SAM 2.1] 마스크 {len(masks)}개, 마스크가 박스를 채운 비율 평균 {np.mean(fill):.0%} "
      f"(나머지는 박스 안의 도로·옆 차) | {ms:.1f} ms")
summary.append(("SAM 2.1", "마스크 (화소마다 차인가)", f"마스크 {len(masks)}개", ms))

# 4. Grounding DINO — 글로 찾기 (09장)
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval().to(device)


def text_to_box(image, text, threshold=0.35):
    inp = gd_proc(images=image, text=f"{text}.", return_tensors="pt").to(device)
    with torch.no_grad():
        o = gd(**inp)
    return gd_proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=threshold, text_threshold=0.25,
                                                          target_sizes=[image.size[::-1]])[0]


g, ms = timed(text_to_box, image, args.text)
g_boxes = g["boxes"].int().tolist()
print(f"\n[4 Grounding DINO] \"{args.text}\" → 박스 {len(g_boxes)}개 | {ms:.1f} ms")
for b, s, lab in zip(g_boxes, g["scores"].tolist(), g["text_labels"]):
    print(f"  {lab} {s:.2f} {b}")
summary.append(("Grounding DINO", "박스 + 글", f"박스 {len(g_boxes)}개", ms))

# 5. CLIP — 사진 한 장을 벡터 하나로, 그 벡터로 COCO 500장에서 비슷한 사진 찾기 (07장)
clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
clip.eval()


def embed(images):
    with torch.no_grad():
        f = clip.encode_image(torch.stack([preprocess(im) for im in images]).to(device)).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()


vec, ms = timed(embed, [image])
print(f"\n[5 CLIP] 사진 → {vec.shape[1]}차원 벡터 [{', '.join(f'{v:.3f}' for v in vec[0, :5])}, ...] | {ms:.1f} ms")
inst = json.loads((COCO_DIR / "instances.json").read_text(encoding="utf-8"))
caps = json.loads((COCO_DIR / "captions.json").read_text(encoding="utf-8"))
first_cap = {}
for c in caps["annotations"]:
    first_cap.setdefault(c["image_id"], c["caption"].strip())
coco = inst["images"]
index = faiss.IndexFlatIP(vec.shape[1])
for i in range(0, len(coco), 50):
    index.add(embed([Image.open(COCO_DIR / "images" / im["file_name"]).convert("RGB") for im in coco[i:i + 50]]))
sim, top = index.search(vec, 3)
print(f"  COCO {index.ntotal}장 중 가장 비슷한 사진 (사람이 쓴 설명):")
for s, i in zip(sim[0], top[0]):
    print(f"  {s:.3f} {coco[i]['file_name']} \"{first_cap[coco[i]['id']]}\"")
summary.append(("CLIP", "벡터 (숫자 512개)", "비슷한 사진 3장", ms))

# 그림 — 앞의 네 방법의 출력. 번호판은 개인정보이므로 Grounding DINO로 찾아 모자이크한다 (16-2)
plates = []
for x1, y1, x2, y2 in boxes:                                                  # 차량 박스마다 잘라서 찾는다 (16-2)
    found = text_to_box(image.crop((x1, y1, x2, y2)), "license plate", 0.2)["boxes"].int().tolist()
    plates += [[a + x1, b + y1, c + x1, d + y1] for a, b, c, d in found if c - a < 0.6 * (x2 - x1)]   # 차 전체를 고른 박스는 뺀다
panels = []
for name in ["OpenCV", "YOLO26s", "SAM 2.1", "Grounding DINO"]:
    v = bgr.copy()
    if name == "OpenCV":
        v[red_mask > 0] = (0, 0, 255)
        for x, y, w, h, _ in blobs:
            cv2.rectangle(v, (x - 8, y - 8), (x + w + 8, y + h + 8), (0, 255, 255), 4)
    if name == "SAM 2.1":
        for k, m in enumerate(masks):
            color = np.array([(60 * k) % 256, 255 - (40 * k) % 256, (90 * k + 120) % 256])
            v[m] = (0.4 * v[m] + 0.6 * color).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:                                              # 가까운 차의 앞유리(운전자 얼굴)는 흐리게
        w, h = x2 - x1, y2 - y1
        if w > 250:
            v[y1 + h // 10:y1 + h // 2, x1 + w // 10:x2 - w // 10] = cv2.GaussianBlur(
                v[y1 + h // 10:y1 + h // 2, x1 + w // 10:x2 - w // 10], (0, 0), 12)
    for x1, y1, x2, y2 in plates:
        w, h = x2 - x1, y2 - y1
        px1, px2, py1, py2 = max(0, x1 - w // 3), min(W, x2 + w // 3), max(0, y1 - h // 2), min(H, y2 + h // 2)
        patch = v[py1:py2, px1:px2]
        v[py1:py2, px1:px2] = cv2.resize(cv2.resize(patch, (max(1, (px2 - px1) // 10), max(1, (py2 - py1) // 10))),
                                         (px2 - px1, py2 - py1), interpolation=cv2.INTER_NEAREST)
    for b in (boxes if name == "YOLO26s" else g_boxes if name == "Grounding DINO" else []):
        cv2.rectangle(v, tuple(b[:2]), tuple(b[2:]), (0, 255, 0) if name == "YOLO26s" else (255, 255, 0), 4)
    crop = cv2.resize(v[200:1080, 300:1620], (660, 440))
    cv2.putText(crop, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    panels.append(crop)
cv2.imwrite(str(OUT / "00_3_four_ways.jpg"), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
print(f"\n그림: 번호판 {len(plates)}개를 찾아 모자이크")

# 6. Qwen3.5 — 사진을 보고 문장으로 (10장). 앞의 모델을 내려 GPU 메모리를 비운다
del sam, gd, clip, yolo
torch.cuda.empty_cache()
proc = AutoProcessor.from_pretrained(args.vlm)
vlm = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16, device_map=device).eval()
messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": "이 사진의 상황을 한국어 두 문장으로 설명하라."}]}]
inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                  return_tensors="pt", enable_thinking=False).to(vlm.device)


def describe(inputs):
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=128, do_sample=False)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


text, ms = timed(describe, inputs)
print(f"\n[6 {args.vlm.split('/')[-1]}] {ms:.0f} ms\n  {text}")
summary.append((args.vlm.split("/")[-1], "문장", f"{len(text)}자", ms))

print("\n방법             출력                       결과            시간")
for name, kind, result, ms in summary:
    print(f"{name:<16} {kind:<24} {result:<14} {ms:8.1f} ms")
print(f"그림: {OUT / '00_3_four_ways.jpg'}")
