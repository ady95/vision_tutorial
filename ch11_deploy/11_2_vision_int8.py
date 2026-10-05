"""11-2. Vision 모델 Quantization — YOLO26s를 FP32 / FP16 / INT8로

같은 YOLO26s를 여러 형식으로 내보내고, 06-2와 같은 COCO val2017 500장(pycocotools)으로 AP를 잰다.
- PyTorch FP32, PyTorch FP16
- ONNX (FP32) → ONNX Runtime CPU (onnxruntime-gpu 1.30은 CUDA 12용이라 CUDA 13 PyTorch 환경에서는 CPU로 쓴다)
- TensorRT FP16, TensorRT INT8 (보정 데이터: coco128 — 시험용 500장과 겹치지 않는 학습 이미지 128장)

실행 (저장소 루트에서, NVIDIA GPU):
    python ch11_deploy/11_2_vision_int8.py
"""
import contextlib
import io
import time
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.converter import coco80_to_coco91_class

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "datasets" / "coco_val500"
W = ROOT / "outputs" / "ch11" / "weights"
W.mkdir(parents=True, exist_ok=True)


def export(fmt, **kw):
    """yolo26s.pt를 fmt 형식으로 내보내고, 결과 파일을 outputs/ch11/weights/에 이름을 붙여 둔다"""
    name = {"onnx": "yolo26s.onnx", "engine": f"yolo26s_{'int8' if kw.get('int8') else 'fp16'}.engine"}[fmt]
    dst = W / name
    if not dst.exists():
        t = time.perf_counter()
        out = Path(YOLO("yolo26s.pt").export(format=fmt, imgsz=640, verbose=False, **kw))
        out.rename(dst)
        print(f"  내보내기 {name}: {time.perf_counter() - t:.0f}초")
    return dst


with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(str(COCO_DIR / "instances.json"))
images = coco.loadImgs(coco.getImgIds())
TO_COCO = coco80_to_coco91_class()

print("내보내기 (처음 한 번만, TensorRT는 이 GPU에 맞춰 엔진을 만든다)")
VARIANTS = [("PyTorch FP32", Path("yolo26s.pt"), 0, {}),
            ("PyTorch FP16", Path("yolo26s.pt"), 0, {"half": True}),
            ("ONNX Runtime CPU", export("onnx"), "cpu", {}),
            ("TensorRT FP16", export("engine", half=True), 0, {}),
            ("TensorRT INT8", export("engine", int8=True, data="coco128.yaml", fraction=1.0), 0, {})]

print(f"\nCOCO val2017 {len(images)}장, 정답 박스 {len(coco.getAnnIds())}개")
print(f"{'형식':<17}{'파일(MB)':>9}{'AP':>7}{'AP50':>7}{'AP_S':>7}{'추론(ms)':>9}{'전체(ms)':>9}")
for name, path, device, kw in VARIANTS:
    model = YOLO(str(path), task="detect")
    for _ in range(5):
        model(COCO_DIR / "images" / images[0]["file_name"], device=device, verbose=False, **kw)    # 워밍업
    dets, infer, total = [], 0.0, 0.0
    for im in images:
        r = model(COCO_DIR / "images" / im["file_name"], conf=0.001, max_det=300, device=device, verbose=False, **kw)[0]
        infer += r.speed["inference"]
        total += sum(r.speed.values())                                         # 전처리 + 추론 + 후처리
        dets += [{"image_id": im["id"], "category_id": TO_COCO[int(c)], "score": float(s),
                  "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
                 for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.evaluate(), ev.accumulate(), ev.summarize()
    s = ev.stats
    print(f"{name:<17}{path.stat().st_size / 1e6:>9.1f}{s[0]:>7.3f}{s[1]:>7.3f}{s[3]:>7.3f}"
          f"{infer / len(images):>9.2f}{total / len(images):>9.2f}")
