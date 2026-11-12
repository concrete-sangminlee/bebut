"""문서에 적힌 명령의 플래그가 실제 스크립트에 있는지 확인한다.

README · docs/*.md · docs/index.html의 `python X.py --flag …` / `python -m pkg.mod --flag …` 줄을 모아,
각 스크립트의 --help에 그 플래그가 있는지 본다. 예전에 문서는 바뀐 플래그·기본값을 따라가지
못했다(예: 인자 없는 `python -m evals.contamination`을 전체 대조처럼 적었다).
"""

import functools
import os
import re
import subprocess
import sys

import pytest

from tests.conftest import ROOT

DOCS = ["README.md", "docs/RESULTS.md", "docs/INFERENCE.md", "docs/OPERATIONS.md", "docs/index.html",
        "CONTRIBUTING.md"]


def doc_commands():
    out = []
    for doc in DOCS:
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read().replace("\\\n", " ")  # 줄 이어쓰기
        for line in text.splitlines():
            for m in re.finditer(r"python3?\s+(-m\s+([\w.]+)|([\w/]+\.py))((?:\s+[^\s`|&;<>#]+)*)", line):
                target = m.group(2) or m.group(3)
                flags = re.findall(r"(?<!\S)(--[a-z][\w-]*)", m.group(4) or "")
                if flags:
                    out.append((doc, target, tuple(flags)))
    return out


@functools.lru_cache(maxsize=None)
def help_text(target):
    if target.endswith(".py"):
        if not os.path.exists(os.path.join(ROOT, target)):
            return None  # llama.cpp의 convert_hf_to_gguf.py 같은 외부 스크립트
        cmd = [sys.executable, target, "--help"]
    else:
        mod = os.path.join(ROOT, *target.split(".")) + ".py"
        if not os.path.exists(mod):
            return None
        cmd = [sys.executable, "-m", target, "--help"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": ROOT})
    return r.stdout + r.stderr if r.returncode == 0 else None


CASES = doc_commands()


def test_docs_have_commands():
    assert len(CASES) > 15  # 추출이 깨지면 아래 검사가 아무것도 안 하게 된다


@pytest.mark.parametrize("doc,target,flags", CASES, ids=[f"{d}:{t}" for d, t, _ in CASES])
def test_documented_flags_exist(doc, target, flags):
    text = help_text(target)
    if text is None:
        pytest.skip(f"{target}: 저장소 밖 스크립트이거나 --help를 돌릴 수 없다(의존성 없음)")
    missing = [f for f in flags if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text)]
    assert not missing, f"{doc}의 `{target}` 명령에 없는 플래그: {missing}"
