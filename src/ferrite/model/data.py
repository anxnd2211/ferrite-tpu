"""Byte-level TinyShakespeare, with a synthetic fallback for offline runs."""

from __future__ import annotations

import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tinyshakespeare(root: str = "data") -> np.ndarray:
    """Return the corpus as uint8 bytes, downloading it (1.1 MB) on first use."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def synthetic(n: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """A learnable fake corpus: a noisy order-2 Markov chain over 64 symbols."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, 64, size=(64, 64, 3))
    out = np.empty(n, dtype=np.uint8)
    out[:2] = rng.integers(0, 64, size=2)
    choice = rng.integers(0, 3, size=n)
    noise = rng.random(n) < 0.05
    rnd = rng.integers(0, 64, size=n)
    for i in range(2, n):
        out[i] = rnd[i] if noise[i] else table[out[i - 2], out[i - 1], choice[i]]
    return out


def load(name: str) -> np.ndarray:
    if name == "tinyshakespeare":
        try:
            return load_tinyshakespeare()
        except Exception as e:  # offline: fall back rather than fail the benchmark
            print(f"[data] download failed ({e}); using synthetic corpus")
    return synthetic()


def split(data: np.ndarray, frac: float = 0.9):
    n = int(len(data) * frac)
    return data[:n], data[n:]


def get_batch(rng: np.random.Generator, data: np.ndarray, batch: int, seq_len: int):
    """Random contiguous windows -> (inputs, targets) as int32 [batch, seq_len]."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch)
    idx = starts[:, None] + np.arange(seq_len + 1)[None, :]
    chunk = data[idx].astype(np.int32)
    return chunk[:, :-1], chunk[:, 1:]
