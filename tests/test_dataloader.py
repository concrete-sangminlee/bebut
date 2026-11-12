"""데이터로더 검증: 예전 random 모드 호환, epoch 모드의 '모든 토큰 정확히 한 번'."""

import numpy as np

from data.dataloader import ShardedDataLoader


def write_shards(d, split, sizes, seed=0):
    rng = np.random.default_rng(seed)
    for i, n in enumerate(sizes):
        rng.integers(0, 60000, n).astype(np.uint16).tofile(d / f"{split}_{i:06d}.bin")


def test_random_mode_matches_legacy_algorithm(tmp_path):
    write_shards(tmp_path, "train", [5000, 3000])
    dl = ShardedDataLoader(str(tmp_path), "train", 4, 32, seed=42, mode="random")
    stream = np.concatenate([np.fromfile(tmp_path / f"train_{i:06d}.bin", np.uint16)
                             for i in range(2)])
    for step in range(5):
        x, y = dl.next_batch()
        starts = np.random.default_rng((42, step)).integers(0, len(stream) - 33, size=4)
        for b, s in enumerate(starts):  # 예전 구현과 같은 위치를 뽑는다 (shard 경계 포함)
            assert (x[b].numpy() == stream[s:s + 32]).all()
            assert (y[b].numpy() == stream[s + 1:s + 33]).all()


def test_legacy_state_without_mode_resumes_as_random(tmp_path):
    write_shards(tmp_path, "train", [4000])
    dl = ShardedDataLoader(str(tmp_path), "train", 2, 16, mode="epoch")
    dl.load_state_dict({"step": 7, "seed": 42})  # mode 키가 없던 시절의 체크포인트
    assert dl.mode == "random" and dl.step == 7


def test_epoch_mode_covers_every_target_once(tmp_path):
    write_shards(tmp_path, "train", [1000, 777])
    T, B = 16, 3
    dl = ShardedDataLoader(str(tmp_path), "train", B, T, seed=1, mode="epoch")
    n = dl.windows_per_epoch()
    shift, _ = dl._epoch_layout(0)
    starts = []
    for _ in range((n + B - 1) // B):
        starts.append(dl._starts())
        dl.pos += B
    starts = np.concatenate(starts)[:n]
    assert len(set(starts.tolist())) == n  # 한 에폭 안에 중복 없음
    targets = np.concatenate([np.arange(s + 1, s + T + 1) for s in starts])
    assert len(set(targets.tolist())) == n * T  # target 위치가 겹치지 않는다
    assert targets.min() == shift + 1 and targets.max() == shift + n * T
    assert dl.total - (n * T + shift) <= 2 * T  # 버리는 꼬리는 윈도우 두 개 이하


def test_epoch_mode_is_deterministic_and_resumable(tmp_path):
    write_shards(tmp_path, "train", [3000])
    a = ShardedDataLoader(str(tmp_path), "train", 4, 32, mode="epoch")
    ref = [a.next_batch()[0] for _ in range(60)]  # 여러 에폭에 걸치게
    b = ShardedDataLoader(str(tmp_path), "train", 4, 32, mode="epoch")
    b.load_state_dict({"step": 37, "seed": 42, "mode": "epoch"})
    for s in range(37, 60):
        assert (b.next_batch()[0] == ref[s]).all()
    # 에폭마다 순서가 다르다
    n_steps = a.windows_per_epoch() // 4
    assert not all((ref[i] == ref[i + n_steps]).all() for i in range(3))


def test_epoch_resume_with_different_batch_size_sees_every_window_once(tmp_path):
    """batch_size를 바꿔 재개해도(configs/base_1b.py가 권하는 조작) 한 에폭의 윈도우를 정확히
    한 번씩 본다. 예전엔 위치를 step × 새 batch_size로 계산해 건너뛰거나 되감았다."""
    write_shards(tmp_path, "train", [4000])
    T = 16
    for b1, b2 in ((2, 4), (4, 2), (3, 5)):
        a = ShardedDataLoader(str(tmp_path), "train", b1, T, mode="epoch")
        n = a.windows_per_epoch()
        seen = [a.next_batch()[1][:, 0].tolist() for _ in range(7)]  # 첫 target 토큰 위치 대신 내용
        state = a.state_dict()
        b = ShardedDataLoader(str(tmp_path), "train", b2, T, mode="epoch")
        b.load_state_dict(state)
        starts = [s for batch in seen for s in batch]
        rest = []
        while len(starts) + len(rest) < n:
            rest += b._starts().tolist()
            b.next_batch()
        layout_shift, perm = b._epoch_layout(0)
        expected = [layout_shift + int(w) * T for w in perm[:n]]
        got = [layout_shift + int(w) * T for w in perm[: 7 * b1]] + rest
        assert got[:n] == expected  # 이어지는 자리가 정확히 다음 윈도우
    # pos가 없던 예전 체크포인트: 당시 batch_size로 위치를 복원한다
    c = ShardedDataLoader(str(tmp_path), "train", 4, T, mode="epoch")
    c.load_state_dict({"step": 10, "seed": 42, "mode": "epoch", "batch_size": 2})
    assert c.pos == 20
