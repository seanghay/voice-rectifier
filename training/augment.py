"""Makes clean recordings sound like real homes: other microphones, rooms, noise and volume."""

import os

import numpy as np

from voice_rectifier.audio import load_audio, prepare_clip


def loudness(x, frame=320):
    """Average power of the louder half of the frames, so silence doesn't count."""
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2) + 1e-10)
    power = np.sort(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1))
    return float(np.mean(power[n // 2:]) + 1e-10)


def change_speed(x, factor):
    t = np.arange(int(len(x) / factor)) * factor
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def add_echo(x, rate, rng):
    """Simulated room: a decaying burst of noise used as the room's echo."""
    rt60 = rng.uniform(0.1, 0.8)
    t = np.arange(int(min(rt60, 0.6) * rate)) / rate
    echo = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    echo[:int(rng.uniform(0.002, 0.02) * rate)] = 0
    echo *= np.sqrt(10 ** (-rng.uniform(-3, 12) / 10) / (np.sum(echo ** 2) + 1e-10))
    echo[0] = 1
    n = 1 << (len(x) + len(echo) - 2).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(echo, n), n)[:len(x)]
    return (y * np.sqrt(loudness(x) / loudness(y))).astype(np.float32)


def change_microphone(x, rate, rng):
    """Random tone colour: bass/treble tilt, a few peaks and dips, cheap-mic band limits."""
    f = np.maximum(np.fft.rfftfreq(len(x), 1 / rate), 1)
    gain_db = rng.uniform(-3, 3) * np.log2(f / 1000)
    for _ in range(rng.integers(0, 4)):
        gain_db += rng.uniform(-8, 8) * np.exp(-0.5 * (np.log2(f / rng.uniform(150, 6000)) / rng.uniform(0.2, 1)) ** 2)
    gain = 10 ** (gain_db / 20)
    if rng.random() < 0.7:
        gain /= np.sqrt(1 + (rng.uniform(60, 400) / f) ** (2 * rng.integers(1, 4)))
    if rng.random() < 0.7:
        gain /= np.sqrt(1 + (f / rng.uniform(3000, 7500)) ** (2 * rng.integers(2, 6)))
    y = np.fft.irfft(np.fft.rfft(x) * gain, len(x))
    return (y * np.sqrt(loudness(x) / loudness(y))).astype(np.float32)


def colored_noise(n, rng):
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.maximum(np.arange(n // 2 + 1), 1)
    y = np.fft.irfft(spectrum / f ** (rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2]) / 2), n)
    return (y / (np.std(y) + 1e-10)).astype(np.float32)


class Augment:
    """Recording -> randomly altered training clip. strength 0..1 scales how often each effect is used."""

    def __init__(self, cfg, noise_dir=None, other_speech=None, use_audiomentations=False):
        self.cfg = cfg
        self.strength = 1.0
        self.other_speech = other_speech
        self.noises = []
        if noise_dir:
            for name in sorted(os.listdir(noise_dir)):
                if name.endswith(".wav"):
                    x = load_audio(os.path.join(noise_dir, name), cfg)
                    if len(x) >= cfg.clip_samples:
                        self.noises.append(x / (np.std(x) + 1e-10))
        self.extra_before, self.extra_after = [], []
        if use_audiomentations:
            import audiomentations as A
            self.extra_before = [(A.PitchShift(-2.5, 2.5, p=1), 0.3),
                                 (A.TimeStretch(0.85, 1.15, leave_length_unchanged=False, p=1), 0.3)]
            self.extra_after = [(A.SevenBandParametricEQ(-8, 8, p=1), 0.25),
                                (A.GainTransition(-12, 6, p=1), 0.2),
                                (A.Mp3Compression(16, 64, p=1), 0.15),
                                (A.TanhDistortion(0.01, 0.3, p=1), 0.1)]

    def __call__(self, x, rng):
        cfg, s = self.cfg, self.strength
        if rng.random() < 0.6 * s:
            x = change_speed(x, rng.uniform(0.88, 1.12))
        for effect, p in self.extra_before:
            if rng.random() < p * s:
                x = effect(samples=x.astype(np.float32), sample_rate=cfg.sample_rate)
        clip = prepare_clip(x, cfg, rng)
        if rng.random() < 0.4 * s:
            clip = add_echo(clip, cfg.sample_rate, rng)
        if rng.random() < 0.5 * s:
            clip = change_microphone(clip, cfg.sample_rate, rng)
        level = loudness(clip)
        if rng.random() < 0.8 * s:
            if self.noises and rng.random() < 0.6:
                noise = self.noises[rng.integers(len(self.noises))]
                start = rng.integers(0, len(noise) - len(clip) + 1)
                noise = noise[start:start + len(clip)]
            else:
                noise = colored_noise(len(clip), rng)
            clip = clip + noise * np.sqrt(level / 10 ** (rng.uniform(0, 30) / 10))
        if self.other_speech is not None and rng.random() < 0.15 * s:
            talk = self.other_speech(rng)
            clip = clip + talk * np.sqrt(level / loudness(talk) / 10 ** (rng.uniform(8, 25) / 10))
        for effect, p in self.extra_after:
            if rng.random() < p * s:
                clip = effect(samples=clip.astype(np.float32), sample_rate=cfg.sample_rate)[:len(clip)]
        clip = clip * np.sqrt(10 ** (rng.uniform(-45, -12) / 10) / loudness(clip))
        if rng.random() < 0.05 * s:
            clip = np.clip(clip, -0.3, 0.3)
        if rng.random() < 0.05 * s:
            clip = np.round(clip * 127) / 127
        return np.clip(clip, -1, 1).astype(np.float32)
