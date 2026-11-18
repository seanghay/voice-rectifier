from collections import deque

import numpy as np


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
