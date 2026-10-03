# 04. Object Detection과 YOLO

도서 페이지: https://wikidocs.net/book/21494 (목차의 04장)

## 설치

```bash
uv pip install -e ".[dl]"
```

YOLO26 가중치(yolo26n/s/m.pt)와 torchvision의 Faster R-CNN·SSD300 가중치는 처음 실행할 때 자동으로 내려받습니다.
Ultralytics는 AGPL-3.0 라이선스입니다 (도서 04-3 참고).

## 실습 데이터

```bash
python data/make_parts.py          # 04-5에서 쓰는 02장 시험 180장
python data/make_parts_det.py      # 04-4·04-5 검출용 합성 데이터 (train 600, val 150, test 150 + data.yaml)
```

04-6은 차량 사진이 필요합니다. 도서의 결과는 저자가 직접 촬영한 사진·영상(개인정보 때문에 배포하지 않음)으로 측정했습니다.
직접 찍은 차량 사진 폴더를 `--images`로 지정해 실행하세요. 결과 숫자는 사진에 따라 달라집니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 04-1 | Object Detection이란 (IoU·NMS 직접 구현) | `04_1_iou_nms.py` | 실측 완료 |
| 04-2 | Detector의 발전 (Faster R-CNN·SSD300·YOLO26 비교) | `04_2_detectors.py` | 실측 완료 |
| 04-3 | Ultralytics YOLO 시작하기 | `04_3_yolo_inference.py` | 실측 완료 |
| 04-4 | Custom Dataset으로 YOLO 학습하기 | `../data/make_parts_det.py`, `04_4_train.py` | 실측 완료 |
| 04-5 | Detection 모델 평가와 Threshold 조정 | `04_5_evaluate.py` | 실측 완료 |
| 04-6 | 실습: 차량과 번호판 검출 | `04_6_vehicle_plate.py`, `04_6_tiling.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch04_detection/04_3_yolo_inference.py`

## 소요 시간 (RTX 3060)

- 04-4 학습: 모델 하나에 약 6.7분 (50에폭, 두 모델 학습)
- 04-5 평가: 약 1분
- 결과 저장 위치는 `outputs/ch04/` (Ultralytics에 상대 경로를 주면 `runs/detect/` 아래로 들어가므로 절대 경로를 씁니다)
