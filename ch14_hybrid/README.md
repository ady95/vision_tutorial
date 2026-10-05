# 14. Hybrid Vision System

도서 페이지: https://wikidocs.net/book/21494 (목차의 14장)

14-1은 앞 장의 조합을 정리한 개념 절이라 예제가 없습니다.

## 설치

```bash
uv pip install -e ".[dl,foundation,vlm]"
```

- 14-2는 12-3의 평가셋(`data/parts_bench`, 12-3을 먼저 실행)과 02-6·04-4·04-5의 기준값·모델을 그대로 씁니다. 얼룩 세트는 처음 실행할 때 `data/parts_stain/`에 만들어집니다
- vLLM 서버가 필요합니다 (10장 README). 14-3은 2B와 9B 서버 두 개를 띄웁니다

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 14-2 | Cascade (Rule → YOLO → VLM), Routing 기준값, 감사 표본 | `14_2_cascade_routing.py` | 실측 완료 |
| 14-3 | Small → Large (Qwen3.5-2B 확신도 → 9B) | `14_3_small_large.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch14_hybrid/14_2_cascade_routing.py --vlm-url http://localhost:8000/v1`

단계별 판정과 두 모델의 답은 `outputs/ch14/`에 저장되어, 다시 실행하면 저장된 것으로 Routing만 다시 계산합니다 (마지막의 실제 실행은 매번 다시 합니다).

## 소요 시간 (RTX 3090)

- 14-2: 약 5분 (처음 실행, 9B로 480장) + 실제 실행 약 20초
- 14-3: 약 15분 (2B·9B로 1,470문항씩) + 실제 실행 약 5분
