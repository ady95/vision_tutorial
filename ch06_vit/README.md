# 06. Vision Transformer

도서 페이지: https://wikidocs.net/book/21494 (목차의 06장)

## 설치

```bash
uv pip install -e ".[dl,foundation]"
```

timm(ResNet-50·DeiT-S·Swin-T), transformers(DETR), Ultralytics(RT-DETR-L, YOLO26s)의 가중치는 처음 실행할 때 자동으로 내려받습니다.

## 실습 데이터

```bash
python data/download_imagenette.py   # 06-1: Imagenette 검증셋 3,925장 (약 330MB, 1분 이내)
python data/download_coco_val.py     # 06-2: COCO val2017 앞 500장 + 정답 (정답 파일 약 240MB, 이미지는 한 장씩 받아 10분 안팎)
python data/download_samples.py      # 06-2 승합차 프레임: 05-5의 교통 영상
```

- Imagenette는 ImageNet의 일부입니다 (ImageNet 이용 약관: 비상업 연구·교육)
- COCO 정답은 CC BY 4.0, 이미지는 Flickr 원저작자의 라이선스를 따릅니다

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 06-1 | CNN의 한계와 Vision Transformer (Patch Embedding 따라가기, 세 모델 분류, CLS Attention) | `06_1_vit_classify.py` | 실측 완료 |
| 06-1 | CNN vs ViT — 조각 섞기·가리기 (Imagenette) | `06_1_cnn_vs_vit.py` | 실측 완료 |
| 06-2 | DETR: Bipartite Matching, DETR 출력, COCO 500장 비교 (DETR·RT-DETR·YOLO26) | `06_2_detr.py` | 실측 완료 |
| 06-2 | 05-5의 승합차 프레임을 네 가지 방식으로 검출 | `06_2_van_frame.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch06_vit/06_1_vit_classify.py`

## 소요 시간 (RTX 3060)

- 06-1 비교: 모델 셋 x 조건 넷, 약 2분
- 06-2 COCO 500장: 세 모델 합쳐 약 1분 + pycocotools 채점
