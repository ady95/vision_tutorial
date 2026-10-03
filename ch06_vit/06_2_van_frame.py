"""06-2. 05-5의 승합차 다시 보기 — 같은 프레임에서 Detector마다 승합차에 박스를 몇 개 내는가

05-5에서 흰색 승합차가 두 번 세어진 프레임(1,381번째)을 네 가지 방식으로 검출한다.
승합차가 있는 영역(x 1000~1425, 박스 중심 y 500~760)의 박스만 출력한다.

실행 (저장소 루트에서, data/download_samples.py 실행 후):
    python ch06_vit/06_2_van_frame.py
"""
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 1380)                          # 1,381번째 프레임 (0부터 센다)
ok, frame = cap.read()
VEHICLES = {"car", "motorcycle", "bus", "truck"}


def near_van(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return 1000 < cx < 1425 and 500 < cy < 760


def show(name, items):
    items = [(c, s, b) for c, s, b in items if c in VEHICLES and near_van(b)]
    print(f"{name:<28}: " + " | ".join(f"{c} {s:.2f} [{', '.join(f'{v:.0f}' for v in b)}]" for c, s, b in items))


proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval().to(device)
im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
with torch.no_grad():
    o = detr(**proc(images=im, return_tensors="pt").to(device))
r = proc.post_process_object_detection(o, threshold=0.25, target_sizes=[im.size[::-1]])[0]
show("DETR (Query마다 클래스 하나)", [(detr.config.id2label[int(l)], float(s), b.tolist()) for s, l, b in zip(r["scores"], r["labels"], r["boxes"])])

for name, model, kw in [("RT-DETR-L", RTDETR("rtdetr-l.pt"), {}),
                        ("YOLO26s (기본: NMS)", YOLO("yolo26s.pt"), {}),
                        ("YOLO26s (nms=False)", YOLO("yolo26s.pt"), {"nms": False})]:
    r = model(frame, conf=0.25, device=device, verbose=False, **kw)[0]
    show(name, [(r.names[int(c)], float(s), b) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls, r.boxes.conf)])
