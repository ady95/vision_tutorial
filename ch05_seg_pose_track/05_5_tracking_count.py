"""05-5. 실습: 차량 Tracking과 출입 차량 Count

영상에서 차량을 추적해 ID를 유지하고, 가로 기준선을 지나간 차량을 방향별로 센다.

실행 (저장소 루트에서):
    python data/download_samples.py                                  # 공개 교통 영상 내려받기
    python ch05_seg_pose_track/05_5_tracking_count.py --video data/videos/traffic.mp4 --line 0.6
    python ch05_seg_pose_track/05_5_tracking_count.py --video 내_영상.mp4 --line 0.6 --span 0.2 1.0 --tracker botsort.yaml
    python ch05_seg_pose_track/05_5_tracking_count.py --video data/videos/traffic.mp4 --save outputs/ch05/05_5_tracking.mp4
"""
import argparse
import time
from collections import Counter, defaultdict

import cv2
import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--line", type=float, default=0.6, help="기준선의 세로 위치 (화면 높이에 대한 비율)")
ap.add_argument("--span", type=float, nargs=2, default=[0.0, 1.0], help="기준선의 가로 범위 (화면 폭에 대한 비율)")
ap.add_argument("--model", default="yolo26s.pt")
ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml 또는 botsort.yaml")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--agnostic", action="store_true", help="클래스가 달라도 크게 겹친 박스는 하나만 남긴다 (car와 truck으로 이중 검출될 때)")
ap.add_argument("--events", default="", help="통과 기록을 저장할 CSV 경로 (정답과 대조할 때)")
ap.add_argument("--save", default="", help="기준선·ID·누적 대수를 그린 영상을 저장할 경로 (.mp4, 저장하는 만큼 느려진다)")
args = ap.parse_args()
device = 0 if torch.cuda.is_available() else "cpu"
VEHICLES = [2, 3, 5, 7]                                     # car, motorcycle, bus, truck
MARGIN = 15                                                 # 기준선 위아래로 이만큼 벗어나야 "그쪽에 있었다"고 본다

model = YOLO(args.model)
first_side = {}                                             # ID별로 처음 확인된 쪽 (-1: 선 위, +1: 선 아래)
counted = {}                                                # 센 ID → 방향
events = []                                                 # (프레임, ID, 방향, 중심 x)
classes = defaultdict(Counter)                              # ID별 클래스 투표 (프레임마다 클래스가 흔들릴 수 있다)
frames_seen = Counter()
n, t0, line_y, span, writer = 0, time.perf_counter(), None, None, None
for r in model.track(args.video, stream=True, persist=True, tracker=args.tracker, classes=VEHICLES,
                     imgsz=args.imgsz, agnostic_nms=args.agnostic, device=device, verbose=False):
    n += 1
    if line_y is None:
        line_y = r.orig_shape[0] * args.line
        span = [r.orig_shape[1] * s for s in args.span]
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
    for tid, cls, (x1, y1, x2, y2) in zip(ids, r.boxes.cls.int().tolist(), r.boxes.xyxy.tolist()):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        classes[tid][r.names[cls]] += 1
        frames_seen[tid] += 1
        if not span[0] <= cx <= span[1]:                   # 기준선(선분)의 가로 범위 밖은 세지 않는다
            continue
        side = -1 if cy < line_y - MARGIN else (1 if cy > line_y + MARGIN else 0)
        if side == 0:
            continue
        if tid not in first_side:
            first_side[tid] = side
        elif side != first_side[tid] and tid not in counted:   # 반대쪽에서 다시 확인되면 한 번만 센다
            counted[tid] = "아래로" if first_side[tid] < 0 else "위로"
            events.append((n, tid, counted[tid], round(cx)))
    if args.save:                                           # 박스·클래스·ID + 기준선 + 지금까지 센 대수
        vis = r.plot(conf=False, line_width=2)
        cv2.line(vis, (int(span[0]), int(line_y)), (int(span[1]), int(line_y)), (0, 0, 255), 3)
        now = Counter(counted.values())
        cv2.putText(vis, f"down {now['아래로']}  up {now['위로']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        if writer is None:
            fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
sec = time.perf_counter() - t0
if writer is not None:
    writer.release()

print(f"{args.video} | {args.model} + {args.tracker} | imgsz {args.imgsz}{' | agnostic NMS' if args.agnostic else ''}")
print(f"{n}프레임, {sec:.1f}초 (초당 {n / sec:.1f}프레임) | 기준선 y={line_y:.0f}, x={span[0]:.0f}~{span[1]:.0f}")
print(f"추적 ID {len(frames_seen)}개 | 10프레임 미만으로 끊긴 ID {sum(v < 10 for v in frames_seen.values())}개")
total = Counter(counted.values())
by_class = Counter((classes[t].most_common(1)[0][0], d) for t, d in counted.items())
print(f"기준선 통과: 아래로 {total['아래로']}대, 위로 {total['위로']}대, 합계 {len(counted)}대")
print("  클래스별:", dict(sorted(by_class.items())))
if args.events:
    with open(args.events, "w", encoding="utf-8") as f:
        f.write("frame,id,direction,class,x\n")
        for frame, tid, d, x in events:
            f.write(f"{frame},{tid},{d},{classes[tid].most_common(1)[0][0]},{x}\n")
