# 16. 실전 프로젝트

도서 페이지: https://wikidocs.net/book/21494 (목차의 16장)

## 설치

프로젝트마다 앞 장의 그룹을 함께 사용합니다. 각 프로젝트 폴더의 README를 참고하세요.

16-4 프로젝트는 OCR 엔진으로 RapidOCR(PaddleOCR 모델을 ONNX Runtime으로 실행)을 사용합니다: `uv pip install rapidocr`
OCR 엔진의 설치·원리·모델 선택은 「OCR 따라하기」(https://wikidocs.net/book/21475)와 예제 저장소 [ocr_tutorial](https://github.com/ady95/ocr_tutorial)에서 다룹니다.

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 16-1 | 제품 불량 검사 — Rule과 Deep Learning 비교 | `p1_defect_inspection/` | 실측 완료 |
| 16-2 | 차량 번호판 인식 시스템 | (안내 페이지 — 예제 없음. 04-6과 「OCR 따라하기」 참고) | - |
| 16-3 | 자연어 이미지 검색·분할 시스템 | `p3_text_search_segment/` | 실측 완료 |
| 16-4 | 산업 Vision Hybrid Pipeline | `p4_hybrid_pipeline/` | 실측 완료 |

실행 방법은 각 프로젝트 폴더의 README에 적습니다.
