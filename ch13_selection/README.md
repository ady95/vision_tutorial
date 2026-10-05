# 13. 어떤 Vision 알고리즘을 선택해야 하는가

도서 페이지: https://wikidocs.net/book/21494 (목차의 13장)

13-1, 13-2는 앞 장의 실측을 모아 정리한 개념 절이라 예제가 없습니다.

## 설치

```bash
uv pip install -e ".[dl,foundation]"
python data/download_coco_val.py     # COCO val2017 앞 500장
```

YOLOE-26s를 처음 실행하면 Ultralytics가 필요한 `clip` 패키지를 자동으로 설치합니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 13-3 | 같은 문제를 YOLO26n·s, YOLOE-26s, Grounding DINO tiny(선택: VLM)로 — 정확도·속도·메모리 | `13_3_compare_methods.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch13_selection/13_3_compare_methods.py`

## 소요 시간

- 약 6분 (RTX 3090 기준, 대부분 Grounding DINO). `--vlm-url`을 주면 VLM이 1시간 넘게 더 걸립니다
