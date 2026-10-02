# 02. Rule 기반 Vision

도서 페이지: https://wikidocs.net/book/21494 (목차의 02장)

## 설치

기본 패키지만 필요합니다: `uv pip install -e .`

## 실습 데이터

02-2·02-3·02-6은 합성 부품 이미지를 씁니다. 먼저 만들어 두세요 (조명 3종 x 60장, 정답 마스크 포함, 시드 고정이라 누구나 같은 이미지를 얻습니다).

```bash
python data/make_parts.py
```

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 02-2 | Threshold, Blur, Morphology | `02_2_threshold_morphology.py` | 실측 완료 |
| 02-3 | Edge, Contour, Shape 분석 | `02_3_edge_contour_shape.py` | 실측 완료 |
| 02-4 | Template Matching과 Feature Matching | `02_4_matching.py` | 실측 완료 |
| 02-5 | Perspective Transform | `02_5_perspective.py` | 실측 완료 |
| 02-6 | 실습: Rule 기반 제품 검사 시스템 | `02_6_rule_inspection.py` | 실측 완료 |

실행은 저장소 루트에서 합니다. 예) `python ch02_rule_based/02_2_threshold_morphology.py`
