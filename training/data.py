"""Datasets are folders of folders: <root>/<phrase>/*.wav, one folder per phrase, many speakers each."""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from voice_rectifier.audio import features, load_audio, prepare_clip


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
