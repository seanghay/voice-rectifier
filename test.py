"""Run with: pytest test.py"""

import json

import numpy as np
import pytest

from voice_rectifier import AudioConfig, Commands, Model, VoiceDetector, is_speech, prepare_clip, read_wav, write_wav

RATE = 16000


def fake_word(seconds=0.5, pitch=150):
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 20))
    return (0.3 * x / np.abs(x).max() * np.hanning(len(t))).astype(np.float32)


def test_wav_round_trip(tmp_path):
    x = 0.5 * np.sin(np.linspace(0, 200, 8000)).astype(np.float32)
    write_wav(str(tmp_path / "a.wav"), x, 8000)
    y, rate = read_wav(str(tmp_path / "a.wav"))
    assert rate == 8000 and np.abs(x - y).max() < 1e-3


def test_float_wav(tmp_path):
    x = (0.25 * np.ones(100)).astype("<f4")
    header = b"RIFF" + (36 + 400).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    header += (3).to_bytes(2, "little") + (1).to_bytes(2, "little") + RATE.to_bytes(4, "little")
    header += (RATE * 4).to_bytes(4, "little") + (4).to_bytes(2, "little") + (32).to_bytes(2, "little")
    (tmp_path / "f.wav").write_bytes(header + b"data" + (400).to_bytes(4, "little") + x.tobytes())
    y, _ = read_wav(str(tmp_path / "f.wav"))
    assert np.allclose(y, 0.25)


def test_clip_is_centred_on_speech():
    cfg = AudioConfig()
    x = np.concatenate([np.zeros(RATE), fake_word(), np.zeros(2 * RATE)]).astype(np.float32)
    clip = prepare_clip(x, cfg)
    assert len(clip) == cfg.clip_samples
    loudest = np.argmax(np.convolve(np.abs(clip), np.ones(800), "same"))
    assert abs(loudest - len(clip) / 2) < 0.15 * RATE


def test_detector_finds_two_words_and_ignores_noise():
    rng = np.random.default_rng(0)
    quiet = 0.001 * rng.standard_normal(RATE).astype(np.float32)
    stream = np.concatenate([quiet, fake_word(), quiet, fake_word(), quiet])
    detector = VoiceDetector()
    found = [u for i in range(0, len(stream), 1600) for u in detector.feed(stream[i:i + 1600])]
    assert len(found) == 2 and is_speech(found[0])[0]
    assert not is_speech(0.3 * rng.standard_normal(RATE).astype(np.float32))[0]


def test_bundled_model_and_commands(tmp_path):
    model = Model("model")
    with open("model/model.json") as f:
        assert json.load(f)["model_id"] == model.model_id
    a = model.embed(np.concatenate([np.zeros(4000), fake_word(0.6, 140), np.zeros(4000)]).astype(np.float32))
    b = model.embed(np.concatenate([np.zeros(4000), fake_word(0.6, 300), np.zeros(4000)]).astype(np.float32))
    assert a.shape == (128,) and abs(np.linalg.norm(a) - 1) < 1e-4
    commands = Commands(str(tmp_path / "c.json"), model.model_id)
    commands.add("low", a, 0.6)
    commands.add("high", b, 0.6)
    commands.save()
    commands = Commands(str(tmp_path / "c.json"), model.model_id)
    assert commands.match(a, 0.6, model, threshold=0.5)[0] == "low"
    assert commands.match(a, 3.5, model, threshold=0.5)[0] is None  # far too long for this command
    with pytest.raises(SystemExit):
        Commands(str(tmp_path / "c.json"), "another-model")


def test_pi_model_matches_training_model():
    torch = pytest.importorskip("torch")
    from train import Net, to_numpy_weights

    net = Net()
    net.train()
    with torch.no_grad():
        for _ in range(3):
            net(torch.randn(8, 40, 248))
    net.eval()
    x = np.random.default_rng(0).standard_normal((4, 40, 248)).astype(np.float32)
    with torch.no_grad():
        reference = net(torch.from_numpy(x)).numpy()
    model = Model.__new__(Model)
    weights = to_numpy_weights(net)
    model.w = {k[:-3] + ".w": weights[k].astype(np.float32) * weights[k[:-1] + "s"].reshape((-1,) + (1,) * (weights[k].ndim - 1))
               for k in weights if k.endswith(".wq")}
    model.w.update({k: v for k, v in weights.items() if k.endswith(".b")})
    assert np.min(np.sum(reference * model.forward(x), axis=1)) > 0.99
