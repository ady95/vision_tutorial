"""05-1. Segmentation 이해하기 — Semantic(DeepLabV3)과 Instance(Mask R-CNN)를 같은 사진으로 비교

두 모델 모두 torchvision의 공개 가중치를 쓴다 (처음 실행할 때 자동으로 내려받는다).

실행 (저장소 루트에서):
    python ch05_seg_pose_track/05_1_segmentation_types.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import detection as tvd
from torchvision.models import segmentation as tvs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
img = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)


def timed(fn):
    fn()                                                  # 워밍업
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


# 1. Semantic Segmentation — 화소마다 클래스 하나 (DeepLabV3, VOC 21종)
weights = tvs.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
deeplab = tvs.deeplabv3_resnet50(weights=weights).eval().to(device)
names = weights.meta["categories"]
norm = weights.transforms()                               # 리사이즈·정규화 (짧은 변 520)
with torch.no_grad():
    out, ms = timed(lambda: deeplab(norm(rgb)[None].to(device))["out"][0])
label = cv2.resize(out.argmax(0).byte().cpu().numpy(), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
print(f"[Semantic] DeepLabV3 {ms:.1f} ms | 출력 {tuple(out.shape)} (클래스 21 x 높이 x 너비)")
for c in np.unique(label):
    print(f"  {names[c]:<12} {(label == c).mean():6.1%} 화소")
person = (label == names.index("person")).astype(np.uint8)
n_blobs, _ = cv2.connectedComponents(person)
print(f"  person 화소의 연결 덩어리: {n_blobs - 1}개 ← 사람이 몇 명인지는 알려 주지 않는다")

# 2. Instance Segmentation — 물체마다 마스크 하나 (Mask R-CNN, COCO 91종)
mweights = tvd.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
maskrcnn = tvd.maskrcnn_resnet50_fpn_v2(weights=mweights).eval().to(device)
mnames = mweights.meta["categories"]
with torch.no_grad():
    pred, ms = timed(lambda: maskrcnn([rgb.to(device)])[0])
keep = pred["scores"] >= 0.5
print(f"\n[Instance] Mask R-CNN {ms:.1f} ms | 신뢰도 0.5 이상 {int(keep.sum())}개")
inst_person = np.zeros_like(person)
for i in torch.where(keep)[0].tolist():
    m = (pred["masks"][i, 0] >= 0.5).cpu().numpy()
    cls = mnames[int(pred["labels"][i])]
    print(f"  #{i} {cls:<8} {float(pred['scores'][i]):.2f} | 마스크 {m.sum():7d} 화소")
    if cls == "person":
        inst_person |= m.astype(np.uint8)

inter = (person & inst_person).sum()
union = (person | inst_person).sum()
print(f"\nSemantic person 영역과 Instance person 마스크 합집합의 IoU: {inter / union:.3f}")

# 3. 그림 — 왼쪽 Semantic(클래스별 색), 오른쪽 Instance(물체별 색)
PALETTE = np.array([(60, 60, 230), (60, 200, 60), (230, 120, 30), (200, 60, 200), (30, 200, 230), (230, 230, 60)])   # BGR
sem_vis = img.copy()
for k, c in enumerate(c for c in np.unique(label) if c):
    sem_vis[label == c] = (0.4 * sem_vis[label == c] + 0.6 * PALETTE[k % len(PALETTE)]).astype(np.uint8)
inst_vis = img.copy()
for k, i in enumerate(torch.where(keep)[0].tolist()):
    m = (pred["masks"][i, 0] >= 0.5).cpu().numpy()
    inst_vis[m] = (0.4 * inst_vis[m] + 0.6 * PALETTE[k % len(PALETTE)]).astype(np.uint8)
cv2.imwrite(str(OUT / "05_1_semantic_vs_instance.jpg"), np.hstack([sem_vis, inst_vis]))
