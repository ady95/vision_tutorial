# 12. Vision 모델 평가

도서 페이지: https://wikidocs.net/book/21494 (목차의 12장)

## 설치

```bash
uv pip install -e ".[dl,foundation,vlm]"
uv pip install motmetrics        # 12-1에서 Tracking 지표를 표준 구현과 대조할 때 (선택)
```

- motmetrics 1.4.0의 `iou_matrix`는 NumPy 2에서 동작하지 않아(`np.asfarray` 제거), 12-1은 IoU 거리를 직접 계산해 넘깁니다
- 12-2는 vLLM 서버가 필요합니다 (10장 README). 채점 모델은 `--judge-url`로 따로 지정할 수 있습니다

## 실습 데이터

```bash
python data/download_coco_val.py     # COCO val2017 앞 500장 + 박스·마스크 + 설명 문장 (12-1, 12-2)
```

12-3은 02~04장 실습을 먼저 마쳐야 합니다. 02-6의 기준값, 03-5의 `outputs/ch03/resnet18_parts.pt`, 04-4의 `outputs/ch04/runs/finetune`을 그대로 씁니다.
평가셋은 처음 실행할 때 `data/parts_bench/`에 만들어집니다 (seed 2031, 조건 7개 x 60장).

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 12-1 | Task별 평가 지표 (mAP의 클래스 평균, 마스크 IoU·Dice, MOTA·IDF1·HOTA) | `12_1_metrics.py` | 실측 완료 |
| 12-2 | VLM 평가 (정답 매칭·규칙·LLM 채점, CHAIR) | `12_2_vlm_eval.py` | 실측 완료 |
| 12-3 | 운영 데이터로 Benchmark (Rule·CNN·YOLO·VLM 비교) | `12_3_benchmark.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch12_evaluation/12_1_metrics.py`

## 소요 시간

- 12-1: 약 3분 (RTX 3060)
- 12-2: 모델마다 약 30~40분 (자세한 설명 150개 + 채점)
- 12-3: Rule·CNN·YOLO 약 2분, VLM 2B 약 1.5분 (RTX 3060) / 9B 약 2분 (RTX 3090)
