# 05. Segmentation · Pose · Tracking

도서 페이지: https://wikidocs.net/book/21494 (목차의 05장)

## 설치

```bash
uv pip install -e ".[dl]"
```

YOLO26 가중치(yolo26s-seg.pt, yolo26s-pose.pt 등)와 torchvision의 DeepLabV3·Mask R-CNN 가중치는 처음 실행할 때 자동으로 내려받습니다.

## 실습 데이터

```bash
python data/download_samples.py    # bus.jpg, coco_cats.jpg, 교통 영상(data/videos/traffic.mp4, 약 43MB)
```

05-4는 01-2에서 만든 `outputs/ch01/01_2_pan.mp4`를 씁니다. 01-2를 먼저 실행하세요.

교통 영상은 Pexels의 "Traffic Flow In The Highway"(Mike Bird, Pexels License)입니다.
라이선스 조건에 따라 저장소에 넣지 않고, 위 스크립트가 원래 주소에서 내려받습니다.
도서의 05-5에는 이 영상의 결과와 함께, 저자가 직접 촬영한 도로 CCTV 영상(배포하지 않음)의 결과도 실려 있습니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 05-1 | Segmentation 이해하기 (DeepLabV3·Mask R-CNN 비교) | `05_1_segmentation_types.py` | 실측 완료 |
| 05-2 | 실습: YOLO Segmentation | `05_2_yolo_seg.py` | 실측 완료 |
| 05-3 | Pose Estimation과 YOLO Pose | `05_3_yolo_pose.py` | 실측 완료 |
| 05-4 | Object Tracking 이해하기 (Optical Flow·Kalman Filter) | `05_4_tracking_basics.py` | 실측 완료 |
| 05-5 | 실습: 차량 Tracking과 출입 차량 Count | `05_5_tracking_count.py`, `05_5_slitscan.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch05_seg_pose_track/05_2_yolo_seg.py`

## 05-5 옵션

| 옵션 | 뜻 |
|---|---|
| `--line 0.6` | 기준선의 세로 위치 (화면 높이에 대한 비율) |
| `--span 0.2 1.0` | 기준선을 화면 폭의 일부(선분)로 제한. 도로 밖 주차 차량을 세지 않게 |
| `--tracker botsort.yaml` | 추적기 선택 (기본 bytetrack.yaml) |
| `--agnostic` | 클래스가 달라도 겹친 박스는 하나만 (car·truck 이중 검출 방지) |
| `--nms-free` | YOLO26의 one-to-one 출력을 NMS 없이 사용 (도서 06-2에서 비교) |
| `--events 경로.csv` | 통과 기록(프레임, ID, 방향, 클래스, x) 저장 — 정답과 대조할 때 |
| `--save 경로.mp4` | 기준선·ID·누적 대수를 그린 결과 영상 저장 |

## 소요 시간 (RTX 3060)

- 05-5 공개 영상(1080p, 1,800프레임): ByteTrack 약 43초, BoT-SORT 약 80초
- 결과 저장 위치는 `outputs/ch05/`
