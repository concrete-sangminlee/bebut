"""shard .bin에서 학습 배치를 뽑는 데이터로더.

설계: 모든 shard를 memmap으로 열어 하나의 거대한 토큰 스트림처럼 취급하고,
매 step 길이 T+1짜리 윈도우를 꺼낸다. 꺼내는 방식은 두 가지다.

  random  무작위 위치에서 복원추출 (1.2B 사전학습에 쓴 방식, 예전 체크포인트 호환용)
  epoch   스트림을 겹치지 않는 윈도우로 자르고, 에폭마다 순서를 섞어 하나씩 꺼낸다

random은 간단하지만 복원추출이라 같은 구간을 여러 번 뽑고 어떤 구간은 끝내 안 뽑는다.
데이터 크기 N 토큰에서 M 토큰을 학습하면 한 번이라도 본 비율은 1 - exp(-M/N)이다.
1.2B 학습의 웹 구간(20.1B 토큰 / 데이터 23.5B)에 대입하면 약 57% — 나머지 43%의
데이터는 한 번도 못 보고, 그만큼을 중복 샘플로 채웠다. epoch 모드는 한 에폭 안에서
모든 토큰을 정확히 한 번씩 target으로 쓴다. 에폭마다 윈도우 경계도 무작위로 밀어서
같은 문맥 절단이 반복되지 않게 한다.

재개(resume)가 공짜인 이유: 두 모드 모두 무작위성이 (seed, step)만으로 결정되므로
체크포인트에 step 수만 저장하면 중단 지점의 데이터 순서를 정확히 재현한다.
"""

import glob
import os

import numpy as np
import torch

MODES = ("random", "epoch")


class ShardedDataLoader:
    def __init__(self, shard_dir: str, split: str, batch_size: int, seq_len: int,
                 seed: int = 42, device: str = "cpu", mode: str = "random"):
        paths = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.bin")))
        assert paths, f"{shard_dir}에 {split} shard 없음 — data.prepare 먼저 실행"
        assert mode in MODES, mode
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.cum[-1])
        self.B, self.T = batch_size, seq_len
        assert self.total > 2 * (self.T + 1), f"{shard_dir}/{split}: 토큰 {self.total}개로는 부족"
        self.seed = seed
        self.step = 0
        self.device = device
        self.mode = mode
        self._perm_epoch, self._perm, self._shift = None, None, 0

    def state_dict(self):
        return {"step": self.step, "seed": self.seed, "mode": self.mode}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.seed = state["seed"]
        # mode가 없는 상태 = 이 키가 생기기 전 체크포인트 → 그때 방식(random) 그대로 이어간다
        self.mode = state.get("mode", "random")

    # ---------------- 윈도우 선택 ----------------

    def _epoch_layout(self, epoch: int):
        """에폭별 (경계 이동량, 윈도우 순열). 마지막 에폭 것만 캐시한다."""
        if self._perm_epoch != epoch:
            rng = np.random.default_rng((self.seed, epoch, 7))
            shift = int(rng.integers(0, self.T))
            # 윈도우 w = [shift + w·T, +T+1). 에폭마다 shift가 달라도 윈도우 수는 같아야
            # 전역 인덱스 k → (에폭, 순번) 대응이 고정된다 — 그래서 최대 shift 기준으로 센다.
            self._perm = rng.permutation(self.windows_per_epoch()).astype(np.int64)
            self._perm_epoch, self._shift = epoch, shift
        return self._shift, self._perm

    def windows_per_epoch(self) -> int:
        return (self.total - self.T) // self.T  # shift ≤ T-1이어도 끝이 total을 넘지 않는다

    def _starts(self, step: int) -> np.ndarray:
        if self.mode == "random":
            rng = np.random.default_rng((self.seed, step))
            return rng.integers(0, self.total - self.T - 1, size=self.B)
        n = self.windows_per_epoch()
        out = np.empty(self.B, dtype=np.int64)
        for i in range(self.B):  # 배치가 에폭 경계에 걸칠 수 있어 원소별로 계산
            k = step * self.B + i
            shift, perm = self._epoch_layout(k // n)
            out[i] = shift + perm[k % n] * self.T
        return out

    def next_batch(self):
        starts = self._starts(self.step)
        self.step += 1
        xs = np.empty((self.B, self.T + 1), dtype=np.int64)
        for i, start in enumerate(starts):
            xs[i] = self._read(int(start), self.T + 1)
        batch = torch.from_numpy(xs)
        x, y = batch[:, :-1], batch[:, 1:]
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def _read(self, start: int, n: int) -> np.ndarray:
        """전역 오프셋 start부터 n개 토큰 — shard 경계에 걸치면 이어붙인다."""
        out = np.empty(n, dtype=np.int64)
        got = 0
        si = int(np.searchsorted(self.cum, start, side="right") - 1)
        off = start - int(self.cum[si])
        while got < n:
            take = min(n - got, int(self.sizes[si]) - off)
            out[got : got + take] = self.shards[si][off : off + take]
            got += take
            si += 1
            off = 0
        return out
