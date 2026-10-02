"""01-2. OpenCV 시작하기 — 읽기·표시·저장, 동영상, 그리기

화면이 없는 서버에서도 동작하도록 결과는 outputs/ch01 에 파일로 저장한다.

실행:
    python ch01_basics/01_2_opencv_io.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 그림을 파일로 저장
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "images" / "bus.jpg"
OUT = ROOT / "outputs" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)

# 1. 읽기 — 경로가 틀려도 예외가 아니라 None 이 돌아온다
missing = cv2.imread("no_such_file.jpg")
print("없는 파일을 읽은 결과:", missing)

color = cv2.imread(str(IMG))                         # 기본: BGR 3채널
gray = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)    # 1채널
print("color:", color.shape, "| gray:", gray.shape)

# 2. 그리기 — 좌표는 (x, y), 색은 (B, G, R)
canvas = color.copy()
cv2.rectangle(canvas, (20, 230), (800, 730), (0, 255, 0), 3)
cv2.circle(canvas, (440, 600), 12, (0, 0, 255), -1)
cv2.putText(canvas, "bus", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

# 3. 저장 — 확장자로 형식이 정해진다. JPEG 품질도 지정할 수 있다
cv2.imwrite(str(OUT / "01_2_draw.jpg"), canvas)
cv2.imwrite(str(OUT / "01_2_draw_q30.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 30])
for name in ["01_2_draw.jpg", "01_2_draw_q30.jpg"]:
    print(f"{name}: {(OUT / name).stat().st_size // 1024} KB")

# 4. 표시 — Matplotlib은 RGB 순서를 기대하므로 BGR을 그대로 넘기면 색이 뒤바뀐다
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].imshow(canvas)
axes[0].set_title("BGR as-is (wrong)")
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title("BGR -> RGB")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "01_2_show_bgr_rgb.png", dpi=100)

# 5. 동영상 쓰기·읽기 — 샘플 영상을 직접 만들어 다시 읽는다
video_path = OUT / "01_2_pan.mp4"
h, w = color.shape[:2]
fps, size = 30, (640, 360)
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
for i in range(90):                                   # 3초 분량: 버스 위를 가로로 훑는다
    x = int((w - 480) * i / 89)
    frame = cv2.resize(color[300:570, x:x + 480], size)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture(str(video_path))
print("\n열림:", cap.isOpened(),
      "| FPS:", cap.get(cv2.CAP_PROP_FPS),
      "| 프레임 수:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "| 크기:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
count = 0
while True:
    ok, frame = cap.read()
    if not ok:                                        # 마지막 프레임 다음에는 ok=False
        break
    count += 1
cap.release()
print("실제로 읽은 프레임:", count)
