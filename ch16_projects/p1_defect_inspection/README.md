# 16-1 프로젝트: 제품 불량 검사 — Rule과 Deep Learning 비교

같은 부품 검사를 세 방식(Rule, CNN 분류, 결함 영역 분할 + 치수 규칙)으로 만들어 12-3의 현장 평가셋으로 비교합니다.

## 준비

02-6, 03-5(학습된 `outputs/ch03/resnet18_parts.pt`), 12-3(평가셋 `data/parts_bench`)을 먼저 실행합니다.

```bash
uv pip install -e ".[dl,foundation]"
```

## 실행 (저장소 루트에서)

```bash
python ch16_projects/p1_defect_inspection/make_data.py         # 결함 마스크가 붙은 학습 데이터 data/parts_seg (train 600, val 150)
python ch16_projects/p1_defect_inspection/train_seg.py         # YOLO26n-seg 50에폭 (RTX 3060 약 9분)
python ch16_projects/p1_defect_inspection/inspect_compare.py   # 세 방식 비교, 결함 영역 IoU, 리포트 outputs/ch16/p1/report.jpg
```
