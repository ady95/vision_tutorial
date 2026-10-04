# 07. Vision Foundation Model

도서 페이지: https://wikidocs.net/book/21494 (목차의 07장)

## 설치

```bash
uv pip install -e ".[dl,foundation]"
```

CLIP(open_clip, OpenAI 공개 가중치), DINOv2·DeiT·Florence-2(transformers), ResNet-50(timm)의 가중치는 처음 실행할 때 자동으로 내려받습니다.
Florence-2는 transformers에 내장된 구현과 `florence-community/Florence-2-base`·`-large` 가중치를 씁니다 (trust_remote_code 불필요).

## 실습 데이터

```bash
python data/download_imagenette.py   # 07-1, 07-3: Imagenette (06장과 같음)
python data/download_coco_val.py     # 07-2, 07-4: COCO val2017 앞 500장 + 정답 박스 + 설명 문장
python data/download_samples.py      # 07-4 승합차 프레임: 05-5의 교통 영상
python data/make_parts.py            # 07-3 부품 시험 180장 (03장과 같음)
python data/make_parts.py --n 300 --seed 2027 --out data/parts_train   # 07-3 부품 학습 900장
```

COCO 이미지는 원저작자마다 라이선스가 달라(다수가 비상업 조건) 도서에는 COCO 사진을 결과 그림으로 싣지 않습니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 07-1 | Foundation Model과 CLIP (유사도, Zero-shot 10종·1000종) | `07_1_clip_zeroshot.py` | 실측 완료 |
| 07-2 | 실습: CLIP 유사 이미지 검색 (faiss, Recall@K, 질의 유형별 정밀도) | `07_2_clip_search.py` | 실측 완료 |
| 07-3 | DINOv2와 Self-Supervised Learning (kNN·소수 라벨·선형 분류, 부품 결함, Attention·PCA) | `07_3_dinov2.py` | 실측 완료 |
| 07-4 | Florence-2: 하나의 모델, 여러 Task (프롬프트 6종, 승합차, COCO 있는 것·없는 것 찾기) | `07_4_florence2.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch07_foundation/07_2_clip_search.py --query "a dog playing with a frisbee"`

## 소요 시간 (RTX 3060)

- 07-1: ViT-B/32 약 10초, ViT-L/14 약 2분 (Imagenette 3,925장)
- 07-3: 특징 세 종류 x (Imagenette 13,394장 + 부품 1,080장) 약 2분
- 07-4: COCO 질문 1,985개 약 11분
