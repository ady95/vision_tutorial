# 부록

도서 페이지: https://wikidocs.net/book/21494 (목차의 부록)

| 페이지 | 제목 | 파일 | 상태 |
|---|---|---|---|
| 부록 E | Vision 모델 선택표와 GPU별 실행 가능한 모델 | `99_5_gpu_memory.py` | 실측 완료 (RTX 3060) |

실행 (저장소 루트에서, data/download_samples.py 실행 후, SAM 3는 08장 README의 접근 승인 필요):

```bash
uv pip install -e ".[dl,foundation]"
python appendix/99_5_gpu_memory.py
```
