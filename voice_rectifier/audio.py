import shutil
import struct
import subprocess
import wave
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
