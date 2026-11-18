"""Make training phrases from recordings where several people read the same text.

  python -m training.prepare_phrases --list readings.csv --out data/ddd --phrases 3000 --val 300

readings.csv has the columns path,text_id,speaker. Each text read by at least --min-speakers
people becomes one phrase. Only the first 0.8-2.5 s of speech is kept (a random length per
phrase): everyone starts the same sentence with the same words, so these act like short
spoken commands. Missing or unreadable files are skipped and reported.
"""

import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from voice_rectifier.audio import AudioConfig, load_audio, speech_region, write_wav

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phrases", type=int, default=0, help="how many phrases for training (0 = all)")
    parser.add_argument("--val", type=int, default=300, help="how many other phrases to keep for validation")
    parser.add_argument("--min-speakers", type=int, default=3)
    parser.add_argument("--min-seconds", type=float, default=0.8)
    parser.add_argument("--max-seconds", type=float, default=2.5)
    args = parser.parse_args()

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
    if args.phrases:
        train = train[:args.phrases]

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


if __name__ == "__main__":
    main()
