"""06-1. 사전학습 ViT로 이미지 분류 — Patch Embedding을 한 단계씩 따라가고, CNN·Swin과 같은 사진으로 비교

세 모델 모두 ImageNet-1k(1,000종)만으로 학습한 공개 가중치이며, 크기(2,200만~2,800만 파라미터)가 비슷하다.
처음 실행할 때 Hugging Face에서 가중치를 자동으로 내려받는다.

실행 (저장소 루트에서):
    python ch06_vit/06_1_vit_classify.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo, create_transform, resolve_data_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ch06"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"ResNet-50 (CNN)": "resnet50.a1_in1k",
          "DeiT-S (ViT)": "deit_small_patch16_224.fb_in1k",
          "Swin-T": "swin_tiny_patch4_window7_224.ms_in1k"}
IMAGES = [ROOT / "data" / "images" / "bus.jpg", ROOT / "data" / "images" / "coco_cats.jpg"]
info = ImageNetInfo()

# 1. 이미지가 토큰이 되기까지 — DeiT-S(ViT)의 앞부분을 한 단계씩
vit = timm.create_model(MODELS["DeiT-S (ViT)"], pretrained=True).eval().to(device)
vit_tf = create_transform(**resolve_data_config({}, model=vit))     # 짧은 변을 줄이고 가운데 224x224를 잘라 정규화
x = vit_tf(Image.open(IMAGES[1]).convert("RGB"))[None].to(device)
print(f"입력 이미지 텐서: {tuple(x.shape)}")
print(f"Patch Embedding 층: {vit.patch_embed.proj}")
with torch.no_grad():
    p = vit.patch_embed.proj(x)                              # 16x16 조각마다 384차원 벡터 하나
    print(f"  Conv 출력: {tuple(p.shape)} (벡터 384개 x 14 x 14 조각)")
    tokens = p.flatten(2).transpose(1, 2)                     # 14x14 격자를 한 줄로 펴서 토큰 196개
    print(f"  토큰으로 펴기: {tuple(tokens.shape)}")
    tokens = torch.cat([vit.cls_token.expand(1, -1, -1), tokens], dim=1)
    print(f"  CLS 토큰 붙이기: {tuple(tokens.shape)}")
    tokens = tokens + vit.pos_embed                           # 위치 정보 더하기 (학습된 값)
    print(f"  Position Embedding 더하기: pos_embed {tuple(vit.pos_embed.shape)} → {tuple(tokens.shape)}")
    feats = vit.norm(vit.blocks(tokens))
    print(f"  Transformer 블록 {len(vit.blocks)}개 통과: {tuple(feats.shape)}")
    logits = vit.head(feats[:, 0])                            # CLS 토큰 하나로 분류
    print(f"  CLS 토큰 → 분류 머리: {tuple(logits.shape)}")
    print(f"  직접 계산한 결과와 vit(x)의 차이: {(logits - vit(x)).abs().max().item():.2e}")

# 2. 같은 사진, 세 모델 — 크기·속도·예측
for name, mid in MODELS.items():
    model = timm.create_model(mid, pretrained=True).eval().to(device)
    tf = create_transform(**resolve_data_config({}, model=model))
    xs = [tf(Image.open(p).convert("RGB"))[None].to(device) for p in IMAGES]
    with torch.no_grad():
        for _ in range(10):                                   # 워밍업 (첫 몇 번은 GPU 준비 시간이 섞인다)
            model(xs[0])
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            model(xs[0])
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 50 * 1000
        print(f"\n{name}: 파라미터 {sum(p.numel() for p in model.parameters()):,} | 1장 {ms:.1f} ms")
        for path, x in zip(IMAGES, xs):
            top = model(x).softmax(-1)[0].topk(3)
            pred = ", ".join(f"{info.index_to_description(int(i)).split(',')[0]} {float(v):.2f}" for v, i in zip(top.values, top.indices))
            print(f"  {path.stem:<10} {pred}")

# 3. CLS 토큰은 어디를 보는가 — 마지막 블록의 Attention
captured = {}
hook = vit.blocks[-1].attn.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))
panels = []
for path in IMAGES:
    x = vit_tf(Image.open(path).convert("RGB"))[None].to(device)
    with torch.no_grad():
        vit(x)
        a = vit.blocks[-1].attn
        B, N, C = captured["x"].shape
        qkv = a.qkv(captured["x"]).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
        att = (a.q_norm(qkv[0]) @ a.k_norm(qkv[1]).transpose(-2, -1) * a.scale).softmax(-1)   # (1, 헤드, 197, 197)
    cls_att = att[0, :, 0, 1:].mean(0).reshape(14, 14).cpu().numpy()                      # CLS → 조각 196개, 헤드 평균
    top = np.sort(cls_att.ravel())[::-1]
    print(f"\n[{path.stem}] CLS Attention: 가장 많이 본 조각 {top[0]:.1%}, 상위 20개 조각(전체의 10%)에 {top[:20].sum():.1%} 집중")
    img = (x[0].permute(1, 2, 0).cpu().numpy() * np.array(vit.pretrained_cfg["std"]) + np.array(vit.pretrained_cfg["mean"]))
    img = cv2.cvtColor((img.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heat = cv2.resize(cls_att / cls_att.max(), (224, 224), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((heat.clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panels.append(np.hstack([img, cv2.addWeighted(img, 0.45, heat, 0.55, 0)]))
hook.remove()
cv2.imwrite(str(OUT / "06_1_cls_attention.jpg"), np.vstack(panels))
print(f"→ {OUT / '06_1_cls_attention.jpg'}")
