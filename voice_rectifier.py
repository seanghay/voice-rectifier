"""voice-rectifier: audio, voice detection, the model and saved commands. Needs only NumPy."""

import json
import os
import shutil
import struct
import subprocess
import time
import wave
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 2.5
    win_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0
    preemphasis: float = 0.97
    log_eps: float = 1e-6
    feat_mean: float = 0.0
    feat_std: float = 1.0
    trim_frame_ms: float = 20.0
    trim_rel_db: float = 35.0
    trim_floor_db: float = 12.0
    trim_pad_ms: float = 80.0

    @property
    def clip_samples(self):
        return int(round(self.clip_seconds * self.sample_rate))

    @property
    def n_frames(self):
        return 1 + (self.clip_samples - self.win_length) // self.hop_length

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


def read_wav(path):
    """Any PCM or float WAV -> (mono float32 samples, sample rate)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:  # extensible format: the real tag is in the sub-format
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path}: broken WAV file")
    tag, channels, rate, bits = fmt
    width = bits // 8
    pcm = pcm[: len(pcm) // (width * channels) * width * channels]
    if tag == 3 and bits in (32, 64):
        x = np.frombuffer(pcm, "<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(pcm, np.uint8).astype(np.float32) - 128) / 128
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / (1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(pcm, "<i4").astype(np.float32) / (1 << 31)
    else:
        raise ValueError(f"{path}: unsupported WAV encoding (format {tag}, {bits} bit)")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return np.clip(x, -1, 1), rate


def write_wav(path, x, sample_rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def resample(x, rate_in, rate_out):
    if rate_in == rate_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * rate_out / rate_in))
    spectrum = np.fft.rfft(x.astype(np.float64))
    bins = n_out // 2 + 1
    spectrum = spectrum[:bins] if bins <= len(spectrum) else np.pad(spectrum, (0, bins - len(spectrum)))
    return (np.fft.irfft(spectrum, n=n_out) * n_out / len(x)).astype(np.float32)


def load_audio(path, cfg):
    x, rate = read_wav(path)
    return resample(x, rate, cfg.sample_rate)


def speech_region(x, cfg):
    """Start and end sample of the part of x that is not silence."""
    frame = int(cfg.sample_rate * cfg.trim_frame_ms / 1000)
    n = max(1, len(x) // frame)
    frames = np.resize(x, (n, frame)) if len(x) < frame else x[: n * frame].reshape(n, frame)
    energy = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    peak, floor = energy.max(), np.percentile(energy, 10)
    threshold = max(peak - cfg.trim_rel_db, floor + cfg.trim_floor_db)
    if threshold >= peak:
        threshold = peak - 3
    active = np.flatnonzero(energy >= threshold)
    pad = int(cfg.sample_rate * cfg.trim_pad_ms / 1000)
    return max(0, active[0] * frame - pad), min(len(x), (active[-1] + 1) * frame + pad)


def speech_seconds(x, cfg):
    start, end = speech_region(x, cfg)
    return (end - start) / cfg.sample_rate


def prepare_clip(x, cfg, rng=None):
    """Cut a fixed-length window around the speech. Training passes rng to place it randomly."""
    start, end = speech_region(x, cfg)
    n = cfg.clip_samples
    lo, hi = (start, end - n) if end - start >= n else (end - n, start)
    offset = (lo + hi) // 2 if rng is None else int(rng.integers(lo, hi + 1))
    clip = np.zeros(n, np.float32)
    a, b = max(offset, 0), min(offset + n, len(x))
    if b > a:
        clip[a - offset:b - offset] = x[a:b]
    return clip


def _mel_filters(cfg):
    def to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    hz = 700 * (10 ** (np.linspace(to_mel(cfg.fmin), to_mel(cfg.fmax), cfg.n_mels + 2) / 2595) - 1)
    freqs = np.linspace(0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1)
    filters = np.zeros((cfg.n_mels, len(freqs)))
    for i in range(cfg.n_mels):
        up = (freqs - hz[i]) / (hz[i + 1] - hz[i])
        down = (hz[i + 2] - freqs) / (hz[i + 2] - hz[i + 1])
        filters[i] = np.maximum(0, np.minimum(up, down))
    return filters.T.astype(np.float32)


_filters_cache = {}


def features(clip, cfg):
    """Fixed-length clip -> normalised log-mel spectrogram (n_mels, n_frames)."""
    x = np.asarray(clip, np.float32)
    x = np.concatenate([x[:1], x[1:] - cfg.preemphasis * x[:-1]])
    n = 1 + (len(x) - cfg.win_length) // cfg.hop_length
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, cfg.win_length), strides=(x.strides[0] * cfg.hop_length, x.strides[0]))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(cfg.win_length) / cfg.win_length)
    spectrum = np.fft.rfft(frames * window.astype(np.float32), n=cfg.n_fft, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    key = (cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    if key not in _filters_cache:
        _filters_cache[key] = _mel_filters(cfg)
    mel = np.log(power @ _filters_cache[key] + cfg.log_eps).T
    return ((mel - cfg.feat_mean) / cfg.feat_std).astype(np.float32)


def microphone(sample_rate, device=None, block_ms=100):
    """Yield blocks of microphone audio forever, using ALSA's arecord."""
    if shutil.which("arecord") is None:
        raise RuntimeError("arecord not found: sudo apt install alsa-utils")
    cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(sample_rate)]
    if device:
        cmd += ["-D", device]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    block = int(sample_rate * block_ms / 1000) * 2
    try:
        while True:
            raw = proc.stdout.read(block)
            if not raw:
                raise RuntimeError("microphone stopped; check --device (see: arecord -l)")
            yield np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    finally:
        proc.terminate()


def _db(frame):
    return float(10 * np.log10(np.mean(frame.astype(np.float64) ** 2) + 1e-10))


class VoiceDetector:
    """Cuts a continuous audio stream into utterances (speech between pauses)."""

    frame_ms = 20
    start_db = 10.0     # louder than the background by this much starts speech
    stop_db = 6.0       # quieter than background + this counts as silence
    release_db = 35.0   # ... and so does anything this far below the utterance's peak
    start_frames = 3
    pause_ms = 500      # a pause this long ends the utterance
    preroll_ms = 200
    max_seconds = 4.0   # longer speech is conversation, not a command
    min_db = -55.0

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.flen = sample_rate * self.frame_ms // 1000
        self.buf = np.zeros(0, np.float32)
        self.history = deque(maxlen=3000 // self.frame_ms)  # last 3 s of frame loudness
        self.floor = None
        self.pre = deque(maxlen=self.preroll_ms // self.frame_ms)
        self.too_long = False
        self._reset()

    def _reset(self):
        self.in_speech, self.seg, self.run, self.silence, self.peak = False, [], 0, 0, -200.0

    def calibrate(self, noise):
        """Measure the room's background level from a stretch of quiet audio."""
        for i in range(0, len(noise) - self.flen + 1, self.flen):
            self.history.append(_db(noise[i:i + self.flen]))
        self.floor = float(np.percentile(self.history, 20))
        return self.floor

    def _emit(self, too_long):
        keep = max(0, self.silence - 150 // self.frame_ms)
        frames = self.seg[:len(self.seg) - keep] if keep else self.seg
        self.too_long = too_long
        self._reset()
        self.pre.clear()
        return np.concatenate(frames)

    def feed(self, samples):
        """Add audio; yields each finished utterance."""
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        n = len(self.buf) // self.flen
        frames, self.buf = self.buf[:n * self.flen].reshape(n, self.flen), self.buf[n * self.flen:]
        for frame in frames:
            e = _db(frame)
            self.history.append(e)
            if self.floor is None and len(self.history) < 10:  # first 200 ms: just listen
                self.pre.append(frame)
                continue
            # background level = a quiet moment of the last 3 s, so long talk can't raise it
            self.floor = float(np.percentile(self.history, 20))
            if not self.in_speech:
                self.pre.append(frame)
                self.run = self.run + 1 if e > max(self.floor + self.start_db, self.min_db) else 0
                if self.run >= self.start_frames:
                    self.in_speech, self.seg = True, list(self.pre)
                    self.peak = max(_db(f) for f in self.seg)
                continue
            self.seg.append(frame)
            self.peak = max(self.peak, e)
            speaking = e > max(self.floor + self.stop_db, self.peak - self.release_db, self.min_db)
            self.silence = 0 if speaking else self.silence + 1
            if self.silence * self.frame_ms >= self.pause_ms:
                yield self._emit(too_long=False)
            elif len(self.seg) * self.frame_ms >= self.max_seconds * 1000:
                yield self._emit(too_long=True)

    def flush(self):
        if self.in_speech and self.seg:
            yield self._emit(too_long=False)


def is_speech(x, sample_rate=16000):
    """Rough check that a sound is a voice, not a click, hiss, fan or tap. Returns (ok, reason)."""
    flen = sample_rate // 50
    n = len(x) // flen
    if n < 2:
        return False, "too short"
    frames = x[:n * flen].reshape(n, flen).astype(np.float64)
    energy = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    loud = energy >= energy.max() - 20
    if loud.sum() * 20 < 80:
        return False, "too short"
    spectrum = np.abs(np.fft.rfft(frames[loud] * np.hanning(flen), axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(flen, 1 / sample_rate)
    band = (freqs >= 100) & (freqs <= 4000)
    flatness = np.mean(np.exp(np.mean(np.log(spectrum[:, band]), axis=1)) / np.mean(spectrum[:, band], axis=1))
    if flatness > 0.30:
        return False, "sounds like noise"
    if spectrum[:, band].sum() / spectrum.sum() < 0.3:
        return False, "not in the voice range"
    return True, ""


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


class Commands:
    """The user's commands, saved in a JSON file. Each command keeps the voice print of every example."""

    def __init__(self, path, model_id):
        self.path = path
        self.data = {"model_id": model_id, "commands": {}}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
            if self.data["commands"] and self.data["model_id"] != model_id:
                raise SystemExit(f"{path} was made with a different model. Run: python3 enroll.py rebuild")
            self.data["model_id"] = model_id

    @property
    def names(self):
        return sorted(n for n, c in self.data["commands"].items() if c["examples"])

    def examples(self, name):
        return self.data["commands"][name]["examples"]

    def add(self, name, embedding, seconds, audio_file=None):
        cmd = self.data["commands"].setdefault(name, {"examples": [], "threshold": None})
        cmd["examples"].append({"embedding": [round(float(v), 6) for v in embedding],
                                "seconds": round(float(seconds), 3), "audio": audio_file,
                                "added": time.strftime("%Y-%m-%d %H:%M:%S")})

    def remove(self, name):
        return self.data["commands"].pop(name, None) is not None

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def prototype(self, name):
        return normalize(np.mean([e["embedding"] for e in self.examples(name)], axis=0))

    def seconds(self, name):
        return float(np.mean([e["seconds"] for e in self.examples(name)]))

    def threshold(self, name, model):
        custom = self.data["commands"][name].get("threshold")
        return custom if custom is not None else model.threshold(len(self.examples(name)), self.seconds(name))

    def match(self, embedding, seconds, model, threshold=None, margin=0.03, length_tolerance=1.6):
        """Find the best command for a voice print. Returns (command or None, details)."""
        names = self.names
        if not names:
            return None, {"reason": "no commands"}
        scores = np.array([self.prototype(n) @ embedding for n in names])
        order = np.argsort(-scores)
        best = names[order[0]]
        score = float(scores[order[0]])
        runner_up = float(scores[order[1]]) if len(names) > 1 else -1.0
        bar = threshold if threshold is not None else self.threshold(best, model)
        lengths = [e["seconds"] for e in self.examples(best)]
        shortest, longest = min(lengths) / length_tolerance - 0.1, max(lengths) * length_tolerance + 0.1
        info = {"best": best, "score": round(score, 3), "threshold": round(bar, 3)}
        if score < bar:
            info["reason"] = "not similar enough"
        elif score - runner_up < margin:
            info["reason"] = f"too close to '{names[order[1]]}'"
        elif length_tolerance and not shortest <= seconds <= longest:
            info["reason"] = f"wrong length ({seconds:.1f}s)"
        else:
            return best, info
        return None, info
