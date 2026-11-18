"""Train the voice-print model.

  python -m training.train --train data/sc/train data/ddd/train --val data/sc/val data/ddd/val \\
                           --noise data/sc/noise --out runs/my-model

Every batch teaches the model that recordings of the same phrase by different people
should get similar voice prints, using 1, 2 or 5 examples per phrase like a real user would.
Validation uses phrases (and ideally speakers) the model never trained on.
"""

import argparse
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # many workers x many threads each is slow

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from voice_rectifier.audio import AudioConfig, features, load_audio, prepare_clip
from voice_rectifier.model import normalize
from training.augment import Augment
from training.data import Clips, Episodes, scan
from training.net import Net


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
            torch.save({"net": net.state_dict(), "audio": cfg.to_dict(), "epoch": epoch, "accuracy": report},
                       os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "accuracy": report}) + "\n")


if __name__ == "__main__":
    main()
