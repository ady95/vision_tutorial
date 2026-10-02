# 10. Vision Language Model

도서 페이지: https://wikidocs.net/book/21494 (목차의 10장)

## 설치

`uv pip install -e ".[vlm]"`

vLLM 서버는 Linux GPU 환경에 별도로 설치합니다: `pip install vllm`

| 모델 | 검증 GPU |
|---|---|
| Qwen3.5-2B, Qwen3.5-4B | RTX 3060 12GB |
| Qwen3.5-9B | RTX 3090 24GB (12GB에서는 11장 양자화 버전 사용) |

## 예제 목록

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 10-2 | Qwen3.5 시작하기 | `10_2_qwen35_hello.py` | 예정 |
| 10-3 | 실습: Captioning과 Visual Question Answering | `10_3_caption_vqa.py` | 예정 |
| 10-4 | 실습: Visual Grounding과 공간 이해 | `10_4_grounding_spatial.py` | 예정 |
| 10-5 | 실습: 관계 이해와 Visual Reasoning | `10_5_reasoning.py` | 예정 |
| 10-6 | 실습: Multi-image와 Video VLM | `10_6_video_vlm.py` | 예정 |
| 10-7 | VLM의 Hallucination과 한계 | `10_7_hallucination.py` | 예정 |

실행은 저장소 루트에서 합니다. 예) `python ch10_vlm/10_2_qwen35_hello.py`
