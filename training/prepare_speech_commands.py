"""Download Google Speech Commands (2.4 GB) and split it for training.

  python -m training.prepare_speech_commands --out data/sc

Makes data/sc/train (25 words), data/sc/val (10 other words, spoken by people not in train)
and data/sc/noise (background noise). Files are linked, not copied.
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
VAL_WORDS = ["backward", "bed", "bird", "dog", "follow", "house", "learn", "off", "tree", "visual"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--archive", help="already downloaded speech_commands_v0.02.tar.gz")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
