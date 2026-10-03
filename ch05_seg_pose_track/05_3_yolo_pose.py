"""05-3. Pose Estimation과 YOLO Pose — 사람의 관절 17개 찾기

실행 (저장소 루트에서):
    python ch05_seg_pose_track/05_3_yolo_pose.py
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "data" / "images" / "bus.jpg"
device = 0 if torch.cuda.is_available() else "cpu"
KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]               # COCO 키포인트 17개

model = YOLO("yolo26s-pose.pt")
model(IMG, device=device, verbose=False)                                     # 워밍업
t = time.perf_counter()
r = model(IMG, device=device, verbose=False)[0]
print(f"사람 {len(r.boxes)}명 | {(time.perf_counter() - t) * 1000:.1f} ms | keypoints.data {tuple(r.keypoints.data.shape)} "
      "(사람 수 x 17 x (x, y, 신뢰도))")


def angle(a, b, c):
    """b를 꼭짓점으로 하는 a-b-c 각도 (도)"""
    v1, v2 = np.array(a) - np.array(b), np.array(c) - np.array(b)
    return math.degrees(math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))


for i, (box, kps) in enumerate(zip(r.boxes.xyxy.tolist(), r.keypoints.data.cpu().numpy())):
    xy, conf = kps[:, :2], kps[:, 2]
    seen = conf >= 0.5
    print(f"\n사람 #{i}: 박스 x {box[0]:.0f}~{box[2]:.0f} | 보이는 관절 {int(seen.sum())}/17")
    hidden = [KP[j] for j in range(17) if not seen[j]]
    if hidden:
        print(f"  안 보이는 관절: {', '.join(hidden)}")
    for side in ["left", "right"]:
        s, e, w = (KP.index(f"{side}_{p}") for p in ["shoulder", "elbow", "wrist"])
        if seen[[s, e, w]].all():
            print(f"  {side:<5} 팔꿈치 각도 {angle(xy[s], xy[e], xy[w]):5.1f}도 (어깨-팔꿈치-손목)")
    hips, knees = [KP.index("left_hip"), KP.index("right_hip")], [KP.index("left_knee"), KP.index("right_knee")]
    if seen[hips + knees].all():
        print(f"  엉덩이→무릎 평균 높이 차 {np.mean(xy[knees, 1] - xy[hips, 1]):.0f}px (서 있으면 양수로 크다)")

cv2.imwrite(str(OUT / "05_3_bus_pose.jpg"), r.plot(boxes=False))
