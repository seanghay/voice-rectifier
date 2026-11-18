import json
import os

import numpy as np

from .audio import AudioConfig, features, prepare_clip

STEM = 32
BLOCKS = [(48, 1), (48, 1), (64, 2), (64, 1), (96, 2), (96, 1), (128, 1)]  # (channels, stride)


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def conv3x3(x, w, b, stride, depthwise):
    n, c, h, wd = x.shape
    ho, wo = (h - 1) // stride + 1, (wd - 1) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    hs, ws = stride * (ho - 1) + 1, stride * (wo - 1) + 1
    taps = [xp[:, :, i:i + hs:stride, j:j + ws:stride] for i in range(3) for j in range(3)]
    if depthwise:
        out = sum(w[None, :, 0, k // 3, k % 3, None, None] * t for k, t in enumerate(taps))
        return out + b[None, :, None, None]
    patches = np.stack(taps, axis=2).reshape(n, c * 9, ho * wo)
    return (np.matmul(w.reshape(len(w), -1), patches) + b[None, :, None]).reshape(n, -1, ho, wo)


def pointwise(x, w, b):
    n, c, h, wd = x.shape
    return (np.matmul(w.reshape(len(w), c), x.reshape(n, c, h * wd)) + b[None, :, None]).reshape(n, -1, h, wd)


class Model:
    """Turns speech into a 128-number voice print. Same phrase -> similar voice prints."""

    def __init__(self, model_dir="model"):
        with open(os.path.join(model_dir, "model.json")) as f:
            meta = json.load(f)
        self.audio = AudioConfig.from_dict(meta["audio"])
        self.thresholds = meta.get("thresholds", [])
        self.model_id = meta["model_id"]
        data = np.load(os.path.join(model_dir, "model.npz"))
        self.w = {}
        for key in data.files:
            if key.endswith(".wq"):  # int8 weights + one scale per output channel
                q, scale = data[key].astype(np.float32), data[key[:-1] + "s"]
                self.w[key[:-3] + ".w"] = q * scale.reshape((-1,) + (1,) * (q.ndim - 1))
            elif key.endswith(".b"):
                self.w[key] = data[key].astype(np.float32)

    def forward(self, feats):
        w = self.w
        x = np.maximum(conv3x3(feats[:, None], w["stem.w"], w["stem.b"], 2, False), 0)
        channels = STEM
        for i, (out, stride) in enumerate(BLOCKS):
            y = np.maximum(conv3x3(x, w[f"b{i}.dw.w"], w[f"b{i}.dw.b"], stride, True), 0)
            y = np.maximum(pointwise(y, w[f"b{i}.pw.w"], w[f"b{i}.pw.b"]), 0)
            x = y + x if stride == 1 and channels == out else y
            channels = out
        x = x.mean(axis=2)
        pooled = np.concatenate([x.mean(axis=2), np.sqrt(x.var(axis=2) + 1e-5)], axis=1)
        return normalize(pooled @ w["fc.w"].T + w["fc.b"])

    def embed_clips(self, clips, batch=32):
        out = [self.forward(np.stack([features(c, self.audio) for c in clips[i:i + batch]]))
               for i in range(0, len(clips), batch)]
        return np.concatenate(out)

    def embed(self, wave):
        return self.embed_clips([prepare_clip(wave, self.audio)])[0]

    def threshold(self, n_examples, seconds=None):
        """Minimum similarity needed to accept a command. Short commands need a higher bar."""
        if not self.thresholds:
            return 0.6
        group = self.thresholds[0]
        if seconds is not None:
            group = next(g for g in self.thresholds if g["max_seconds"] is None or seconds < g["max_seconds"])
        by_count = {int(k): v for k, v in group["by_examples"].items()}
        usable = [k for k in by_count if k <= n_examples]
        return by_count[max(usable)] if usable else by_count[min(by_count)]
