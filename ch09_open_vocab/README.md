# 09. Open-Vocabulary Vision

도서 페이지: https://wikidocs.net/book/21494 (목차의 09장)

## 설치

```bash
uv pip install -e ".[dl,foundation]"
```

Grounding DINO(IDEA-Research/grounding-dino-tiny·base), SAM 2.1, SAM 3는 transformers 구현을 씁니다.
YOLOE(yoloe-26s-seg.pt)를 처음 실행하면 Ultralytics가 필요한 `clip` 패키지를 자동으로 설치합니다.
09-3의 SAM 3는 08장 README의 접근 승인 절차가 필요합니다.

## 실습 데이터

```bash
python data/download_coco_val.py     # COCO val2017 앞 500장 + 정답 박스·마스크
python data/download_samples.py      # 05-5의 교통 영상 (1,449번째 프레임)
python data/make_parts_det.py        # 09-2 자동 라벨링: 04-4의 합성 부품 검출 데이터
```

자동 라벨은 `data/parts_det_auto/`에, 그 라벨로 학습한 모델은 `outputs/ch09/runs/auto/`에 저장됩니다.
04-4를 먼저 실행해 두면(`outputs/ch04/runs/finetune`) 사람 라벨로 학습한 모델과 함께 비교합니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 09-2 | Grounding DINO로 텍스트 객체 찾기 (표현 비교, COCO Box Threshold 0.25·0.35·0.5, YOLOE) | `09_2_grounding_dino.py` | 실측 완료 |
| 09-2 | 라벨 초안 자동 생성 → 그 라벨로 YOLO26n 학습 | `09_2_auto_label.py` | 실측 완료 |
| 09-3 | 자연어 기반 객체 분할 (Grounding DINO + SAM 2.1 vs SAM 3, COCO 마스크 채점) | `09_3_text_to_mask.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch09_open_vocab/09_2_grounding_dino.py`

## 소요 시간 (RTX 3060)

- 09-2 COCO 비교: 약 23분 (Grounding DINO는 질문마다 한 번, YOLOE는 12초)
- 09-2 자동 라벨링: 라벨 약 6분 + 학습 약 7분
- 09-3: 약 35분
