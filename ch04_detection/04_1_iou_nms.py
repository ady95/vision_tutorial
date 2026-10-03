"""04-1. Object Detection이란 — IoU와 NMS를 직접 계산한다

YOLO26의 날 출력(후보 8,400개)을 꺼내, 신뢰도로 거르고 NMS로 중복을 지워 최종 박스가 되는 과정을 따라간다.

실행 (저장소 루트에서):
    python ch04_detection/04_1_iou_nms.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "images" / "bus.jpg"


# 1. IoU — 두 박스가 얼마나 겹치는가 (교집합 넓이 / 합집합 넓이)
def iou(a, b):
    """a, b: [x1, y1, x2, y2]"""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


gt = [100, 100, 200, 200]
for name, box in [("같은 박스", [100, 100, 200, 200]), ("오른쪽으로 10px", [110, 100, 210, 200]),
                  ("오른쪽으로 50px", [150, 100, 250, 200]), ("가로세로 2배", [75, 75, 225, 225]),
                  ("겹치지 않음", [300, 300, 400, 400])]:
    print(f"IoU {name:<14}: {iou(gt, box):.3f}")

# 2. YOLO의 날 출력 — NMS 전의 후보들
model = YOLO("yolo26n.pt")
net = model.model.eval()
img = cv2.imread(str(IMG))
h, w = img.shape[:2]
scale = 640 / max(h, w)                                   # 01-4의 Letterbox: 비율 유지 + 회색 여백
nh, nw = round(h * scale), round(w * scale)
top, left = (640 - nh) // 2, (640 - nw) // 2
canvas = np.full((640, 640, 3), 114, np.uint8)
canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
x = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None]
with torch.no_grad():
    pred, extra = net(x)                                  # pred: (1, 84, 8400) = 박스 4 + 클래스 80 점수
pred = pred[0].T                                          # (8400, 84)
boxes_xywh, scores = pred[:, :4], pred[:, 4:]
conf, cls = scores.max(dim=1)
print(f"\n입력 640x640 → 후보 {len(pred)}개 (격자 80x80 + 40x40 + 20x20 = {80 * 80 + 40 * 40 + 20 * 20})")

keep = conf >= 0.25
print(f"신뢰도 0.25 이상 후보: {int(keep.sum())}개 → 클래스별", {
    model.names[c]: int((cls[keep] == c).sum()) for c in cls[keep].unique().tolist()})
b = boxes_xywh[keep]
xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], dim=1)
c_keep, s_keep = cls[keep], conf[keep]


# 3. NMS — 점수가 높은 박스부터 고르고, 그 박스와 많이 겹치는(IoU 큰) 같은 클래스 박스를 지운다
def nms(boxes, scores, classes, iou_th):
    order = scores.argsort(descending=True).tolist()
    picked = []
    while order:
        i = order.pop(0)
        picked.append(i)
        order = [j for j in order if classes[j] != classes[i] or iou(boxes[i].tolist(), boxes[j].tolist()) < iou_th]
    return picked


for th in [0.3, 0.5, 0.7, 0.9, 1.0]:
    print(f"NMS IoU 기준 {th}: {len(nms(xyxy, s_keep, c_keep, th))}개 남음")

mine = nms(xyxy, s_keep, c_keep, 0.7)
ref = torchvision.ops.batched_nms(xyxy, s_keep, c_keep, 0.7).tolist()
print(f"\n직접 만든 NMS와 torchvision NMS의 결과가 같은가: {sorted(mine) == sorted(ref)}")


def to_original(bx):
    """Letterbox 좌표 → 원본 이미지 좌표 (여백을 빼고 배율로 나눈다)"""
    return [(bx[0] - left) / scale, (bx[1] - top) / scale, (bx[2] - left) / scale, (bx[3] - top) / scale]


ours = [(model.names[int(c_keep[i])], to_original(xyxy[i].tolist())) for i in mine]
r = model(IMG, verbose=False)[0]
print(f"Ultralytics 최종 결과 {len(r.boxes)}개와 비교:")
for box, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()):
    best = max((iou(box, o) for n, o in ours if n == model.names[c]), default=0)
    print(f"  {model.names[c]:<8} 가장 비슷한 직접 계산 박스와 IoU {best:.3f}")

# 4. YOLO26의 one-to-one 출력 — 처음부터 물체 하나에 후보 하나만 높은 점수를 주도록 학습되었다
o2o = extra["one2one"]["scores"][0].sigmoid().max(dim=0).values
print(f"\none-to-one 출력에서 신뢰도 0.25 이상 후보: {int((o2o >= 0.25).sum())}개 (NMS 없이 바로 최종 결과)")
