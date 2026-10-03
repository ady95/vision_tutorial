"""05-4. Object Tracking 이해하기 — Optical Flow와 Kalman Filter

1) Optical Flow: 01-2에서 만든 영상(버스 사진을 일정한 속도로 훑음)에서 움직임을 재고 정답과 비교한다
2) Kalman Filter: 흔들리고 가끔 사라지는 검출 결과에서 물체의 위치를 추정한다

실행 (저장소 루트에서, 01-2 실행 후):
    python ch05_seg_pose_track/05_4_tracking_basics.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "outputs" / "ch01" / "01_2_pan.mp4"

# 01-2의 영상: 810px 사진에서 480px 폭을 89프레임 동안 330px 옮기며 잘라 640px로 키웠다
true_dx = -(810 - 480) / 89 * (640 / 480)              # 화면 속 내용은 왼쪽으로 움직인다
print(f"정답: 프레임당 {true_dx:.2f}px (가로 방향)")

# 1. Optical Flow — 희소(Lucas-Kanade)와 조밀(Farneback)
cap = cv2.VideoCapture(str(VIDEO))
ok, prev = cap.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
sparse, dense = [], []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=8)   # 추적하기 좋은 모서리 점
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
    sparse.append(np.median((p1 - p0)[st.ravel() == 1][:, 0, 0]))
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)          # 화소마다 (dx, dy)
    dense.append(np.median(flow[..., 0]))
    prev = gray
cap.release()
for name, v in [("Lucas-Kanade (점 200개)", sparse), ("Farneback (모든 화소)", dense)]:
    v = np.array(v)
    print(f"{name:<22}: 평균 {v.mean():6.2f}px | 표준편차 {v.std():.2f} | 정답과의 오차 {abs(v.mean() - true_dx):.2f}px")

# 2. Kalman Filter — 등속으로 움직이는 차량을, 흔들리고(표준편차 8px) 20%는 놓치는 검출로 따라간다
rng = np.random.default_rng(0)
T = 100
true_x = 10 + 5.0 * np.arange(T)                         # 프레임당 5px씩 이동
detected = true_x + rng.normal(0, 8, T)
missing = rng.random(T) < 0.2
detected[missing] = np.nan

kf = cv2.KalmanFilter(2, 1)                              # 상태 (위치, 속도), 관측 (위치)
kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0]], np.float32)
kf.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
kf.measurementNoiseCov = np.array([[64]], np.float32)   # 검출의 흔들림 8px의 제곱
kf.statePost = np.array([[detected[0]], [0]], np.float32)
kf.errorCovPost = np.eye(2, dtype=np.float32) * 100
est = []
for z in detected:
    pred = kf.predict()                                  # 1단계: 속도로 다음 위치를 예측
    if not np.isnan(z):
        pred = kf.correct(np.array([[z]], np.float32))   # 2단계: 검출이 있으면 예측과 섞어 고친다
    est.append(float(pred[0, 0]))
est = np.array(est)

ok = ~missing
print(f"\n검출 누락 {missing.sum()}프레임 / {T}프레임")
print(f"검출 그대로의 오차(RMSE)   : {np.sqrt(np.nanmean((detected - true_x) ** 2)):.2f}px (누락 프레임은 위치 없음)")
print(f"Kalman 추정의 오차(RMSE)   : {np.sqrt(np.mean((est[20:] - true_x[20:]) ** 2)):.2f}px (앞 20프레임 제외)")
print(f"  그중 누락 프레임에서만    : {np.sqrt(np.mean((est[20:][missing[20:]] - true_x[20:][missing[20:]]) ** 2)):.2f}px")
print(f"추정한 속도: 프레임당 {float(kf.statePost[1, 0]):.2f}px (정답 5.00)")
