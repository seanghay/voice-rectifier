"""Build training data: folders of <phrase>/*.wav, each phrase spoken by several people.

  python prepare.py speech-commands --out data/sc
      Downloads Google Speech Commands (2.4 GB). Makes data/sc/train (25 words), data/sc/val
      (10 other words, by speakers not in train) and data/sc/noise. Files are linked, not copied.

  python prepare.py phrases --list readings.csv --out data/phrases --count 3000
      From any recordings where several people read the same text. readings.csv has the columns
      path,text_id,speaker. Keeps the first 0.8-2.5 s of speech of each recording (a random length
      per phrase), so each text becomes a short command-like phrase.
"""

import argparse
import csv
import hashlib
import os
import tarfile
import urllib.request
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from voice_rectifier import AudioConfig, load_audio, speech_region, write_wav

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
VAL_WORDS = ["backward", "bed", "bird", "dog", "follow", "house", "learn", "off", "tree", "visual"]


def speech_commands(args):
    raw = os.path.join(args.out, "raw")
    if not os.path.isdir(os.path.join(raw, "yes")):
        archive = args.archive or os.path.join(args.out, "speech_commands_v0.02.tar.gz")
        os.makedirs(args.out, exist_ok=True)
        if not os.path.exists(archive):
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive)
        print("extracting")
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")

    for word in sorted(os.listdir(raw)):
        if word.startswith("_") or not os.path.isdir(os.path.join(raw, word)):
            continue
        split = "val" if word in VAL_WORDS else "train"
        os.makedirs(os.path.join(args.out, split, word), exist_ok=True)
        for name in os.listdir(os.path.join(raw, word)):
            speaker = name.split("_")[0]
            held_out = int(hashlib.sha1(speaker.encode()).hexdigest()[:8], 16) % 100 < 20
            if held_out == (split == "val"):  # 20% of speakers are only ever heard in val
                dst = os.path.join(args.out, split, word, name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(os.path.join(raw, word, name)), dst)

    os.makedirs(os.path.join(args.out, "noise"), exist_ok=True)
    for name in os.listdir(os.path.join(raw, "_background_noise_")):
        dst = os.path.join(args.out, "noise", name)
        if name.endswith(".wav") and not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(raw, "_background_noise_", name)), dst)
    for split in ("train", "val"):
        folders = os.listdir(os.path.join(args.out, split))
        count = sum(len(os.listdir(os.path.join(args.out, split, w))) for w in folders)
        print(f"{split}: {len(folders)} words, {count} recordings")



CFG = AudioConfig()


def convert(job):
    src, dst, seconds = job
    try:
        x = load_audio(src, CFG)
    except Exception as e:
        return f"{src}: {e}"
    start, end = speech_region(x, CFG)
    x = x[start:min(end, start + int(seconds * CFG.sample_rate))]
    if len(x) < 0.3 * CFG.sample_rate:
        return f"{src}: almost no speech"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    write_wav(dst, x / max(1.0, np.abs(x).max()), CFG.sample_rate)
    return None


def phrases(args):
    readings, missing = defaultdict(list), 0
    with open(args.list, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if os.path.isfile(row["path"]):
                readings[row["text_id"]].append((row["path"], row["speaker"]))
            else:
                missing += 1
    texts = sorted(t for t, r in readings.items() if len({s for _, s in r}) >= args.min_speakers)
    print(f"{missing} files missing; {len(texts)} texts read by {args.min_speakers}+ people")
    rng = np.random.default_rng(0)
    rng.shuffle(texts)
    val, train = texts[:args.val], texts[args.val:]
    if args.count:
        train = train[:args.count]

    jobs = []
    for split, chosen in (("train", train), ("val", val)):
        for t in chosen:
            seconds = rng.uniform(args.min_seconds, args.max_seconds)
            folder = os.path.join(args.out, split, "".join(c if c.isalnum() or c in "-_" else "_" for c in t))
            jobs += [(p, os.path.join(folder, f"{speaker}_{i}.wav"), seconds)
                     for i, (p, speaker) in enumerate(readings[t])]
    with Pool() as pool:
        errors = [e for e in pool.imap_unordered(convert, jobs, chunksize=16) if e]
    print(f"{len(train)} training phrases, {len(val)} validation phrases, "
          f"{len(jobs) - len(errors)} recordings written, {len(errors)} failed")
    for e in errors[:10]:
        print("  " + e)



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    p = sub.add_parser("speech-commands")
    p.add_argument("--out", required=True)
    p.add_argument("--archive", help="already downloaded speech_commands_v0.02.tar.gz")
    p = sub.add_parser("phrases")
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=0, help="how many phrases for training (0 = all)")
    p.add_argument("--val", type=int, default=300, help="how many other phrases to keep for validation")
    p.add_argument("--min-speakers", type=int, default=3)
    p.add_argument("--min-seconds", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=2.5)
    args = parser.parse_args()
    speech_commands(args) if args.what == "speech-commands" else phrases(args)


if __name__ == "__main__":
    main()
