"""11-4. ONNX, TensorRT와 실시간 Edge Vision — Latency, Throughput, FPS

1) Latency: 사진 한 장(bus.jpg)을 200번 — 형식마다 전처리 / 추론 / 후처리 시간 (중앙값, 95% 지점)
2) Throughput: 16장을 한 번에(배치 16) 넣을 때 초당 처리 장수
3) FPS: 05-5의 교통 영상 300프레임을 읽어 검출까지 — 영상 읽기를 포함한 실제 속도

11_2_vision_int8.py가 만든 ONNX·TensorRT 파일을 쓴다 (없으면 먼저 실행).

실행 (저장소 루트에서):
    python ch11_deploy/11_4_onnx_tensorrt.py              # NVIDIA GPU가 있는 컴퓨터
    python ch11_deploy/11_4_onnx_tensorrt.py --cpu-only   # GPU 없는 컴퓨터 (PyTorch CPU, ONNX Runtime CPU)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "outputs" / "ch11" / "weights"
ap = argparse.ArgumentParser()
ap.add_argument("--cpu-only", action="store_true")
args = ap.parse_args()
img = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))

if args.cpu_only:
    VARIANTS = [("PyTorch CPU", "yolo26s.pt", "cpu", {}), ("ONNX Runtime CPU", W / "yolo26s.onnx", "cpu", {})]
else:
    VARIANTS = [("PyTorch FP32", "yolo26s.pt", 0, {}), ("PyTorch FP16", "yolo26s.pt", 0, {"half": True}),
                ("ONNX Runtime CPU", W / "yolo26s.onnx", "cpu", {}),
                ("TensorRT FP16", W / "yolo26s_fp16.engine", 0, {}), ("TensorRT INT8", W / "yolo26s_int8.engine", 0, {})]

# 1. Latency — 한 장씩
print(f"{'형식':<18}{'전처리':>8}{'추론 중앙값':>11}{'추론 95%':>9}{'후처리':>8}{'합계':>8}  (ms, 200회)")
for name, path, device, kw in VARIANTS:
    model = YOLO(str(path), task="detect")
    for _ in range(10):
        model(img, device=device, verbose=False, **kw)                        # 워밍업
    sp = np.array([list(model(img, device=device, verbose=False, **kw)[0].speed.values()) for _ in range(200)])
    pre, inf, post = sp[:, 0], sp[:, 1], sp[:, 2]
    print(f"{name:<18}{np.median(pre):>8.2f}{np.median(inf):>11.2f}{np.percentile(inf, 95):>9.2f}{np.median(post):>8.2f}"
          f"{np.median(sp.sum(1)):>8.2f}")

# 2. Throughput — 배치 16 (TensorRT는 배치 16 엔진을 따로 만든다)
if not args.cpu_only:
    b16 = W / "yolo26s_fp16_b16.engine"
    if not b16.exists():
        Path(YOLO("yolo26s.pt").export(format="engine", half=True, batch=16, imgsz=640, verbose=False)).rename(b16)
    batch = [img] * 16
    print(f"\n{'형식':<18}{'배치 1 (장/초)':>14}{'배치 16 (장/초)':>15}")
    for name, path, kw in [("PyTorch FP16", "yolo26s.pt", {"half": True}), ("TensorRT FP16", None, {})]:
        rates = []
        for bs in [1, 16]:
            p = path or (W / "yolo26s_fp16.engine" if bs == 1 else b16)
            model = YOLO(str(p), task="detect")
            src = batch[:bs]
            for _ in range(5):
                model(src, device=0, verbose=False, **kw)
            t = time.perf_counter()
            for _ in range(20):
                model(src, device=0, verbose=False, **kw)
            rates.append(20 * bs / (time.perf_counter() - t))
        print(f"{name:<18}{rates[0]:>14.0f}{rates[1]:>15.0f}")

# 3. FPS — 영상 읽기부터 검출까지 (1920x1080, 300프레임)
print(f"\n{'형식':<18}{'FPS':>6}  (영상 읽기 + 검출, 300프레임)")
for name, path, device, kw in VARIANTS:
    model = YOLO(str(path), task="detect")
    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
    model(img, device=device, verbose=False, **kw)
    n, t = 0, time.perf_counter()
    while n < 300:
        ok, frame = cap.read()
        if not ok:
            break
        model(frame, device=device, verbose=False, **kw)
        n += 1
    print(f"{name:<18}{n / (time.perf_counter() - t):>6.1f}")
cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "traffic.mp4"))
n, t = 0, time.perf_counter()
while n < 300 and cap.read()[0]:
    n += 1
print(f"{'(영상 읽기만)':<18}{n / (time.perf_counter() - t):>6.1f}")
