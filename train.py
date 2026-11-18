"""Train the model. Writes model.npz, model.json and checkpoint.pt to --out.

  python train.py --train data/sc/train --val data/sc/val --noise data/sc/noise --out runs/new

Datasets are folders of <phrase>/*.wav, each phrase spoken by several people. Every batch
teaches the model that the same phrase from different people should give similar embeddings,
with 1, 2 or 5 examples per phrase. Recordings are randomly altered to sound like other
microphones, rooms, noise and volumes. Validation uses phrases not seen in training.
"""

import argparse
import hashlib
import json
import math
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")  # many workers x many threads each is slow

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from voice_rectifier import BLOCKS, STEM, AudioConfig, Model, features, load_audio, normalize, prepare_clip

class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = ConvBN(cin, cin, 3, stride, groups=cin)
        self.pw = ConvBN(cin, cout, 1)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return y + x if self.residual else y


class Net(nn.Module):
    """PyTorch twin of voice_rectifier.model.Model, used for training."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(1, STEM, 3, stride=2)
        blocks, cin = [], STEM
        for cout, stride in BLOCKS:
            blocks.append(Block(cin, cout, stride))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.pool_bn = nn.BatchNorm1d(2 * cin)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2 * cin, 128)

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        pooled = torch.cat([x.mean(dim=2), torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)], dim=1)
        return F.normalize(self.fc(self.dropout(self.pool_bn(pooled))), dim=1)


def _fold(conv_bn):
    bn = conv_bn.bn
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    w = conv_bn.conv.weight.detach().double() * scale.view(-1, 1, 1, 1)
    b = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    return w.numpy(), b.numpy()


def to_numpy_weights(net):
    """Fold batch-norm into the layers and store weights as int8 (one scale per output channel)."""
    net = net.cpu().eval()
    layers = {"stem": _fold(net.stem)}
    for i, block in enumerate(net.blocks):
        layers[f"b{i}.dw"] = _fold(block.dw)
        layers[f"b{i}.pw"] = _fold(block.pw)
    bn = net.pool_bn
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    shift = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    w, b = net.fc.weight.detach().double(), net.fc.bias.detach().double()
    layers["fc"] = ((w * scale[None]).numpy(), (b + w @ shift).numpy())
    out = {}
    for name, (w, b) in layers.items():
        flat = w.reshape(len(w), -1)
        scale = np.maximum(np.abs(flat).max(axis=1), 1e-12) / 127
        out[name + ".wq"] = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        out[name + ".ws"] = scale.astype(np.float32)
        out[name + ".b"] = b.astype(np.float32)
    return out


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


def scan(roots, min_per_phrase=2, max_per_phrase=None, seed=0):
    """-> (phrase names, [(wav path, phrase index)], [which root each phrase came from])"""
    rng = np.random.default_rng(seed)
    phrases, items, source = [], [], []
    for r, root in enumerate(roots):
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav"))
            if len(files) < min_per_phrase:
                continue
            if max_per_phrase and len(files) > max_per_phrase:
                files = [files[i] for i in sorted(rng.choice(len(files), max_per_phrase, replace=False))]
            items += [(f, len(phrases)) for f in files]
            phrases.append(name)
            source.append(r)
    return phrases, items, source


class Clips(Dataset):
    def __init__(self, items, cfg, augment=None, seed=0):
        self.items, self.cfg, self.augment, self.seed = items, cfg, augment, seed
        self.pid, self.rng = None, None

    def __len__(self):
        return len(self.items)

    def random_clip(self, rng):
        return prepare_clip(load_audio(self.items[rng.integers(len(self.items))][0], self.cfg), self.cfg, rng)

    def __getitem__(self, i):
        if self.pid != os.getpid():  # each data-loader worker needs its own random numbers
            self.pid = os.getpid()
            self.rng = np.random.default_rng([self.seed, torch.initial_seed() % 2 ** 32, self.pid])
            seed = int(self.rng.integers(2 ** 31))
            np.random.seed(seed)
            random.seed(seed)
        path, label = self.items[i]
        x = load_audio(path, self.cfg)
        clip = self.augment(x, self.rng) if self.augment else prepare_clip(x, self.cfg)
        return torch.from_numpy(features(clip, self.cfg)), label


class Episodes(Sampler):
    """Each batch: n_way random phrases x per_phrase clips each. Every dataset gets an equal share."""

    def __init__(self, items, source, n_way, per_phrase, n_batches, seed=0):
        self.by_phrase = {}
        for i, (_, label) in enumerate(items):
            self.by_phrase.setdefault(label, []).append(i)
        self.phrases = sorted(self.by_phrase)
        counts = np.bincount([source[p] for p in self.phrases])
        weights = np.array([1 / counts[source[p]] for p in self.phrases])
        self.p = weights / weights.sum()
        self.n_way, self.per_phrase, self.n_batches = min(n_way, len(self.phrases)), per_phrase, n_batches
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for p in self.rng.choice(self.phrases, self.n_way, replace=False, p=self.p):
                pool = self.by_phrase[p]
                batch += self.rng.choice(pool, self.per_phrase, replace=len(pool) < self.per_phrase).tolist()
            yield batch


def accuracy(emb, labels, shots, n_way=10, episodes=600, seed=1):
    """How often the right phrase is picked among n_way phrases, given `shots` examples each."""
    rng = np.random.default_rng(seed)
    by_phrase = {p: np.flatnonzero(labels == p) for p in np.unique(labels)}
    phrases = [p for p, idx in by_phrase.items() if len(idx) > shots]
    correct = []
    for _ in range(episodes):
        chosen = rng.choice(phrases, min(n_way, len(phrases)), replace=False)
        protos, queries, answers = [], [], []
        for j, p in enumerate(chosen):
            idx = rng.permutation(by_phrase[p])
            protos.append(normalize(emb[idx[:shots]].mean(0)))
            queries += list(idx[shots:shots + 5])
            answers += [j] * len(idx[shots:shots + 5])
        correct += list(np.argmax(emb[queries] @ np.stack(protos).T, axis=1) == answers)
    return float(np.mean(correct))


def spec_augment(x):
    b, n_mels, n_frames = x.shape
    for size, axis, width in ((n_mels, 1, 6), (n_frames, 2, 15), (n_mels, 1, 6), (n_frames, 2, 15)):
        w = torch.randint(0, width + 1, (b, 1), device=x.device)
        start = (torch.rand(b, 1, device=x.device) * (size - w)).long()
        pos = torch.arange(size, device=x.device)[None]
        mask = (pos >= start) & (pos < start + w)
        x = x.masked_fill(mask[:, :, None] if axis == 1 else mask[:, None, :], 0)
    return x


@torch.no_grad()
def embed_all(net, items, cfg, device, workers):
    net.eval()
    loader = DataLoader(Clips(items, cfg), batch_size=256, num_workers=workers)
    out = np.concatenate([net(x.to(device)).cpu().numpy() for x, _ in loader])
    net.train()
    return out


def save(net, cfg, out, check):
    """Write the checkpoint and the int8 model the Pi uses, and check both give the same embeddings."""
    torch.save({"net": net.state_dict(), "audio": cfg.to_dict()}, os.path.join(out, "checkpoint.pt"))
    np.savez(os.path.join(out, "model.npz"), **to_numpy_weights(net))
    with open(os.path.join(out, "model.npz"), "rb") as f:
        model_id = hashlib.sha1(f.read()).hexdigest()[:16]
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump({"model_id": model_id, "audio": cfg.to_dict(), "thresholds": []}, f, indent=1)
    with torch.no_grad():
        reference = net.cpu()(torch.from_numpy(check)).numpy()
    similarity = np.sum(reference * Model(out).forward(check), axis=1).min()
    if similarity < 0.95:
        raise SystemExit(f"int8 model differs from the trained model (similarity {similarity:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="dataset folder(s)")
    parser.add_argument("--val", nargs="+", required=True, help="dataset folder(s) with unseen phrases")
    parser.add_argument("--noise", help="folder of background-noise .wav files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batches", type=int, default=300, help="batches per epoch")
    parser.add_argument("--phrases-per-batch", type=int, default=20)
    parser.add_argument("--clips-per-phrase", type=int, default=8)
    parser.add_argument("--clip-seconds", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--audiomentations", action="store_true", help="extra effects from the audiomentations library")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count()))
    args = parser.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AudioConfig(clip_seconds=args.clip_seconds)

    phrases, items, source = scan(args.train)
    val_sets = []
    for folder in args.val:
        _, val_items, _ = scan([folder], max_per_phrase=300)
        labels = np.array([label for _, label in val_items])
        fewest = np.bincount(labels).min()
        val_sets.append((folder, val_items, labels, [k for k in (1, 2, 3, 5) if k < fewest]))
    print(f"training on {len(phrases)} phrases, {len(items)} recordings, device {device}")

    sample = np.random.default_rng(0).choice(len(items), min(2000, len(items)), replace=False)
    cfg.feat_mean, cfg.feat_std = 0.0, 1.0
    raw = np.stack([features(prepare_clip(load_audio(items[i][0], cfg), cfg), cfg) for i in sample])
    cfg.feat_mean, cfg.feat_std = float(raw.mean()), float(raw.std())

    data = Clips(items, cfg)
    data.augment = Augment(cfg, args.noise, other_speech=data.random_clip, use_audiomentations=args.audiomentations)
    sampler = Episodes(items, source, args.phrases_per_batch, args.clips_per_phrase, args.batches)
    n_way, m = sampler.n_way, args.clips_per_phrase

    net = Net().to(device)
    log_scale = torch.nn.Parameter(torch.tensor(math.log(10.0), device=device))
    class_w = torch.nn.Parameter(torch.randn(len(phrases), 128, device=device) * 0.01)
    params = list(net.parameters()) + [log_scale, class_w]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.batches, pct_start=0.05)

    rng = np.random.default_rng(0)
    first_val = val_sets[0][1]
    check = np.stack([features(prepare_clip(load_audio(first_val[i][0], cfg), cfg), cfg)
                      for i in rng.choice(len(first_val), min(200, len(first_val)), replace=False)])

    best = -1
    for epoch in range(1, args.epochs + 1):
        data.augment.strength = min(1.0, (epoch - 1) / 4)  # start easy, add effects over 4 epochs
        loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers, prefetch_factor=4)
        for x, y in loader:
            emb = net(spec_augment(x.to(device)))
            y = y.to(device)
            # match each clip to the average of k other clips of its phrase, for k = 1, 2, 5
            e = emb.view(n_way, m, -1)
            scale = log_scale.exp().clamp(1, 100)
            loss = 0
            for k in (1, 2, 5):
                perm = torch.randperm(m, device=device)
                protos = F.normalize(e[:, perm[:k]].mean(1), dim=-1)
                logits = scale * e[:, perm[k:]].reshape(-1, e.shape[-1]) @ protos.T
                loss = loss + F.cross_entropy(logits, torch.arange(n_way, device=device).repeat_interleave(m - k)) / 3
            # and to its own phrase among all training phrases (with a margin)
            cos = emb @ F.normalize(class_w, dim=1).T
            loss = loss + 0.5 * F.cross_entropy(30 * (cos - 0.2 * F.one_hot(y, len(phrases))), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            opt.step()
            sched.step()

        report, scores = {}, []
        for folder, val_items, labels, shots in val_sets:
            emb = embed_all(net, val_items, cfg, device, args.workers)
            report[folder] = {k: accuracy(emb, labels, k) for k in shots}
            scores += list(report[folder].values())
        score = float(np.mean(scores))
        print(f"epoch {epoch}: loss {loss.item():.3f}  " + "  ".join(
            f"{os.path.basename(os.path.dirname(f.rstrip('/')))}: " +
            " ".join(f"{k}-shot {100 * a:.1f}%" for k, a in r.items()) for f, r in report.items()), flush=True)
        if score > best:
            best = score
            net.eval()
            save(net, cfg, args.out, check)
            net.to(device).train()
            print(f"  saved to {args.out}")
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "accuracy": report}) + "\n")


if __name__ == "__main__":
    main()
