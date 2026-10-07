"""부록 E. 이 책의 Vision 모델을 GPU에 하나씩 올려 메모리를 잰다

- 가중치: 불러온 직후 늘어난 메모리
- 한 장 최대: bus.jpg 한 장을 추론하는 동안의 최대 메모리
둘 다 PyTorch가 잡은 메모리(torch.cuda)만 센다. nvidia-smi로 보면 CUDA 컨텍스트 몇백 MB가 더해진다.
정밀도는 본문과 같다 (Florence-2만 FP16, 나머지는 FP32).

실행 (저장소 루트에서, data/download_samples.py 실행 후):
    uv pip install -e ".[dl,foundation]"
    python appendix/99_5_gpu_memory.py
"""
import gc
from pathlib import Path

import open_clip
import timm
import torch
from PIL import Image
from transformers import (AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, DetrForObjectDetection,
                          DetrImageProcessor, Florence2ForConditionalGeneration, Sam2Model, Sam2Processor, Sam3Model,
                          Sam3Processor, SamModel, SamProcessor)
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
BUS = ROOT / "data" / "images" / "bus.jpg"
image = Image.open(BUS).convert("RGB")
BOX = [[[50, 400, 250, 900]]]                                          # 왼쪽 사람을 감싸는 박스 (SAM 프롬프트)
dev = "cuda"


def measure(name, load, run):
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    m = load()
    weight = torch.cuda.memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        run(m)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    print(f"{name:<26}{weight / 2**30:>9.2f}{peak / 2**30:>11.2f}")
    del m


def yolo(path, cls=YOLO):
    return lambda: cls(path).to(dev)


def yolo_run(m):
    m.predict(str(BUS), device=0, verbose=False)


def hf(model_cls, mid, proc_cls=AutoProcessor, **kw):
    return lambda: (proc_cls.from_pretrained(mid), model_cls.from_pretrained(mid, **kw).eval().to(dev))


def timm_model(mid):
    return lambda: timm.create_model(mid, pretrained=True).eval().to(dev)


def timm_run(m):
    m(torch.randn(1, 3, 224, 224, device=dev))


def clip_model(arch):
    return lambda: open_clip.create_model_and_transforms(arch, pretrained="openai", device=dev)


def clip_run(m):
    model, _, preprocess = m
    model.encode_image(preprocess(image)[None].to(dev))


def detr_run(m):
    proc, model = m
    model(**proc(images=image, return_tensors="pt").to(dev))


def dino_run(m):
    proc, model = m
    model(**proc(images=image, return_tensors="pt").to(dev))


def florence_run(m):
    proc, model = m
    inp = proc(text="<OD>", images=image, return_tensors="pt").to(dev, torch.float16)
    model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=256, num_beams=3)


def sam_run(m):
    proc, model = m
    model(**proc(images=image, input_boxes=BOX, return_tensors="pt").to(dev), multimask_output=False)


def sam2_run(m):
    proc, model = m
    inp = proc(images=image, input_boxes=BOX, return_tensors="pt").to(dev)
    model(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"], multimask_output=False)


def sam3_run(m):
    proc, model = m
    model(**proc(images=image, text="person", return_tensors="pt").to(dev))


def gdino_run(m):
    proc, model = m
    model(**proc(images=image, text="bus. person.", return_tensors="pt").to(dev))


print(f"GPU: {torch.cuda.get_device_name(0)} | 사진: bus.jpg 810x1080 한 장")
print(f"{'모델':<26}{'가중치(GiB)':>9}{'한 장 최대(GiB)':>11}")
measure("YOLO26n (04장)", yolo("yolo26n.pt"), yolo_run)
measure("YOLO26s (04장)", yolo("yolo26s.pt"), yolo_run)
measure("YOLO26s-seg (05-2)", yolo("yolo26s-seg.pt"), yolo_run)
measure("YOLO26s-pose (05-3)", yolo("yolo26s-pose.pt"), yolo_run)
measure("RT-DETR-L (06-2)", yolo("rtdetr-l.pt", RTDETR), yolo_run)
measure("DETR R50 (06-2)", hf(DetrForObjectDetection, "facebook/detr-resnet-50", DetrImageProcessor), detr_run)
measure("ResNet-50 (06-1)", timm_model("resnet50.a1_in1k"), timm_run)
measure("DeiT-S (06-1)", timm_model("deit_small_patch16_224.fb_in1k"), timm_run)
measure("CLIP ViT-B/32 (07-2)", clip_model("ViT-B-32"), clip_run)
measure("CLIP ViT-L/14 (07-2)", clip_model("ViT-L-14"), clip_run)
measure("DINOv2-S (07-3)", hf(AutoModel, "facebook/dinov2-small"), dino_run)
measure("Florence-2 base FP16 (07-4)", hf(Florence2ForConditionalGeneration, "florence-community/Florence-2-base", dtype=torch.float16), florence_run)
measure("Florence-2 large FP16 (07-4)", hf(Florence2ForConditionalGeneration, "florence-community/Florence-2-large", dtype=torch.float16), florence_run)
measure("SAM ViT-B (08-2)", hf(SamModel, "facebook/sam-vit-base", SamProcessor), sam_run)
measure("SAM 2.1 Hiera-B+ (08-3)", hf(Sam2Model, "facebook/sam2.1-hiera-base-plus", Sam2Processor), sam2_run)
measure("SAM 3 (08-4)", hf(Sam3Model, "facebook/sam3", Sam3Processor), sam3_run)
measure("Grounding DINO tiny (09-2)", hf(AutoModelForZeroShotObjectDetection, "IDEA-Research/grounding-dino-tiny"), gdino_run)
measure("Grounding DINO base (09-2)", hf(AutoModelForZeroShotObjectDetection, "IDEA-Research/grounding-dino-base"), gdino_run)
