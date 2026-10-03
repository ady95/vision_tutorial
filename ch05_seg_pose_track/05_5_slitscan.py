"""05-5. 정답 만들기 — 슬릿 스캔(Slit-scan)으로 기준선을 지나간 차량을 사람이 센다

기준선 위치의 화소 줄(높이 3px)을 프레임마다 잘라 위에서 아래로 쌓으면, 가로는 화면의 x, 세로는 시간인 이미지가 된다.
서 있는 물체는 세로 줄무늬로, 선을 지나간 물체는 짧은 덩어리로 보인다. 배경(시간 중앙값)을 빼면 지나간 것만 남는다.

실행 (저장소 루트에서):
    python ch05_seg_pose_track/05_5_slitscan.py --video data/videos/traffic.mp4 --line 0.6
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--line", type=float, default=0.6)
ap.add_argument("--rows", type=int, default=600, help="그림 한 장에 담을 시간 줄 수")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
strips = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    y = int(frame.shape[0] * args.line)
    strips.append(frame[y - 1:y + 2].mean(axis=0).astype(np.uint8))   # 3줄 평균 → 1줄
cap.release()
scan = np.stack(strips)                                               # (프레임 수, 너비, 3)
background = np.median(scan, axis=0).astype(np.uint8)                 # 시간 중앙값 = 서 있는 배경
moving = np.abs(scan.astype(np.int16) - background).max(axis=2)        # 배경과 다른 정도 (cv2.absdiff는 브로드캐스팅 불가)
print(f"프레임 {len(scan)}개 → 슬릿 스캔 {scan.shape[1]}x{scan.shape[0]}")

for i in range(0, len(scan), args.rows):
    part = scan[i:i + args.rows].copy()
    part[moving[i:i + args.rows] < 40] //= 3                           # 움직임이 없는 곳은 어둡게
    vis = cv2.resize(part, (960, part.shape[0]), interpolation=cv2.INTER_AREA)
    for t in range(0, part.shape[0], 50):                              # 50프레임마다 눈금
        cv2.putText(vis, str(i + t), (2, t + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(str(OUT / f"05_5_slitscan_{i:05d}.png"), vis)
print(f"→ {OUT}/05_5_slitscan_*.png (가로 = 화면 x, 세로 = 프레임 번호)")
