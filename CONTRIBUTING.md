# 기여 가이드

개인 학습 프로젝트지만, 버그 제보나 개선 제안은 언제든 환영합니다.

## 이슈

- 버그는 재현 방법과 환경(OS, Python, PyTorch, GPU)을 함께 적어주세요.
- 기능 제안은 어떤 문제를 해결하는지 먼저 설명해주세요.

## PR

1. 변경 전 [README](README.md)와 [docs/OPERATIONS.md](docs/OPERATIONS.md)로 구조를 파악해주세요.
2. 코드 스타일은 기존 파일을 따라주세요 (docstring에 "왜"를 적는 방식, 한국어 주석).
3. 제출 전 최소한 아래 두 검증은 통과해야 합니다 (CPU, 1분 내):

   ```bash
   python -m tokenizer.test_bpe
   python test_model.py
   ```

4. 학습 루프나 데이터 파이프라인을 바꿨다면 `configs/debug_30m` 같은 작은 설정으로
   실제 실행이 되는지 확인해주세요 (GPU 없이도 가능).

## 범위에 대해

이 저장소는 "GPU 한 장으로 밑바닥부터 학습"이라는 제약을 의도적으로 유지합니다.
`transformers`, `accelerate`, 분산학습 프레임워크 등 무거운 의존성을 추가하는 PR은
프로젝트 취지와 맞지 않아 받지 않을 가능성이 높습니다.
