"""Pretend to be a home and set how strict the model should be.

  python -m training.calibrate --phrases data/sc/val data/ddd/val --background background.txt --write

Many times over, it picks 5 unseen phrases as "commands" (with 1, 2 or 3 examples each), then:
  - plays other people saying those phrases     -> how often the command is caught
  - plays an hour of ordinary talk (TV, radio)  -> how often something fires by mistake
For each command length and number of examples it finds the lowest accept level that keeps
mistakes to at most --target per hour, and with --write saves it into model/model.json.

--background is a folder of .wav files or a text file with one .wav path per line.
"""

import argparse
import json
import os

import numpy as np

from voice_rectifier.audio import load_audio, prepare_clip, speech_seconds
from voice_rectifier.model import Model, normalize
from voice_rectifier.vad import VoiceDetector, is_speech
from training.data import scan

LENGTH_GROUPS = [0.8, 1.3, None]  # commands shorter than 0.8 s, 0.8-1.3 s, longer
LEVELS = np.round(np.arange(0.40, 0.951, 0.005), 3)


def background_talk(model, paths, hours):
    """Stream background audio through the same detector as listen.py; keep what reaches matching."""
    cfg, rate = model.audio, model.audio.sample_rate
    detector = VoiceDetector(rate)
    rng = np.random.default_rng(0)
    clips, lengths, total = [], [], 0.0
    for path in paths:
        if total >= hours * 3600:
            break
        x = load_audio(path, cfg)
        gap = (0.002 * rng.standard_normal(int(rng.uniform(0.2, 1.5) * rate))).astype(np.float32)
        audio = np.concatenate([gap, x])
        total += len(audio) / rate
        for i in range(0, len(audio), rate // 10):
            for utterance in detector.feed(audio[i:i + rate // 10]):
                if not detector.too_long and is_speech(utterance, rate)[0]:
                    clips.append(prepare_clip(utterance, cfg))
                    lengths.append(speech_seconds(utterance, cfg))
    return model.embed_clips(clips), np.array(lengths), total / 3600


def simulate(emb, lengths, labels, phrases, bg_emb, bg_len, hours, shots, trials, rng):
    """-> (share of commands caught, mistakes per hour) for every accept level."""
    caught, total, mistakes = np.zeros(len(LEVELS)), 0, np.zeros(len(LEVELS))
    for _ in range(trials):
        chosen = rng.choice(phrases, 5, replace=False)
        protos, shortest, longest, tests, answers = [], [], [], [], []
        for j, p in enumerate(chosen):
            idx = rng.permutation(np.flatnonzero(labels == p))
            protos.append(normalize(emb[idx[:shots]].mean(0)))
            shortest.append(lengths[idx[:shots]].min() / 1.6 - 0.1)
            longest.append(lengths[idx[:shots]].max() * 1.6 + 0.1)
            tests += list(idx[shots:])
            answers += [j] * (len(idx) - shots)
        protos, shortest, longest = np.stack(protos), np.array(shortest), np.array(longest)
        for x, length, answer in ((emb[tests], lengths[tests], np.array(answers)), (bg_emb, bg_len, None)):
            scores = x @ protos.T  # same rules as Commands.match
            order = np.argsort(-scores, axis=1)
            rows = np.arange(len(x))
            best, pick = scores[rows, order[:, 0]], order[:, 0]
            ok = ((best - scores[rows, order[:, 1]] >= 0.03) & (length >= shortest[pick]) & (length <= longest[pick]))
            accepted = ok[None] & (best[None] >= LEVELS[:, None])
            if answer is None:
                mistakes += accepted.sum(1)
            else:
                caught += (accepted & (pick == answer)[None]).sum(1)
                total += len(answer)
    return caught / total, mistakes / trials / hours


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model")
    parser.add_argument("--phrases", nargs="+", required=True, help="dataset folder(s) of unseen phrases")
    parser.add_argument("--background", required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--target", type=float, default=1.0, help="allowed mistakes per hour")
    parser.add_argument("--write", action="store_true", help="save the result into the model")
    args = parser.parse_args()

    model = Model(args.model)
    cfg = model.audio
    if os.path.isdir(args.background):
        paths = sorted(os.path.join(args.background, f) for f in os.listdir(args.background) if f.endswith(".wav"))
    else:
        with open(args.background) as f:
            paths = [line.strip() for line in f if line.strip()]
    missing = [p for p in paths if not os.path.isfile(p)]
    paths = [p for p in paths if os.path.isfile(p)]
    np.random.default_rng(0).shuffle(paths)
    print(f"background: {len(paths)} files ({len(missing)} missing, skipped)")
    bg_emb, bg_len, hours = background_talk(model, paths, args.hours)
    print(f"  {hours:.2f} hours -> {len(bg_len)} bits of talk reach the matching step")

    _, items, _ = scan(args.phrases, max_per_phrase=60)
    waves = [load_audio(p, cfg) for p, _ in items]
    emb = model.embed_clips([prepare_clip(w, cfg) for w in waves])
    lengths = np.array([speech_seconds(w, cfg) for w in waves])
    labels = np.array([label for _, label in items])
    phrase_len = {p: lengths[labels == p].mean() for p in np.unique(labels)}

    groups, low = [], 0.0
    for high in LENGTH_GROUPS:
        in_group = [p for p, s in phrase_len.items() if s >= low and (high is None or s < high)]
        name = f"{low}-{high}s" if high else f"{low}s+"
        found = {}
        for shots in (1, 2, 3):
            usable = [p for p in in_group if np.sum(labels == p) > shots]
            if len(usable) < 10:
                continue
            caught, mistakes = simulate(emb, lengths, labels, usable, bg_emb, bg_len, hours,
                                        shots, args.trials, np.random.default_rng(shots))
            ok = np.flatnonzero(mistakes <= args.target)
            if len(ok):
                i = ok[0]
                found[str(shots)] = float(LEVELS[i])
                print(f"commands {name:9} {shots} example(s): accept level {LEVELS[i]:.3f} "
                      f"-> {100 * caught[i]:.1f}% caught, {mistakes[i]:.2f} mistakes/hour  ({len(usable)} phrases)")
        if found:
            groups.append({"max_seconds": high, "by_examples": found})
        low = high

    if groups:
        groups[-1]["max_seconds"] = None  # the last group covers everything longer
    if args.write and groups:
        path = os.path.join(args.model, "model.json")
        with open(path) as f:
            meta = json.load(f)
        meta["thresholds"] = groups
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)
        print(f"saved to {path}")


if __name__ == "__main__":
    main()
