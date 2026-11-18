"""Turn a training checkpoint into the small model the Pi uses, and check it still gives the same answers.

  python -m training.export runs/my-model/best.pt --check data/sc/val --out model
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from voice_rectifier.audio import AudioConfig, features, load_audio, prepare_clip
from voice_rectifier.model import Model
from training.data import scan
from training.net import Net, to_numpy_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--check", required=True, help="dataset folder with recordings to compare on")
    parser.add_argument("--out", default="model")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = AudioConfig.from_dict(ckpt["audio"])
    net = Net()
    net.load_state_dict(ckpt["net"])
    net.eval()

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "model.npz")
    np.savez(npz, **to_numpy_weights(net))
    with open(npz, "rb") as f:
        model_id = hashlib.sha1(f.read()).hexdigest()[:16]
    meta_path = os.path.join(args.out, "model.json")
    thresholds = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            thresholds = json.load(f).get("thresholds", [])
        print("kept the old thresholds; run training.calibrate to update them for this model")
    with open(meta_path, "w") as f:
        json.dump({"model_id": model_id, "audio": cfg.to_dict(), "thresholds": thresholds}, f, indent=1)

    _, items, _ = scan([args.check], max_per_phrase=30)
    feats = np.stack([features(prepare_clip(load_audio(p, cfg), cfg), cfg) for p, _ in items])
    with torch.no_grad():
        reference = net(torch.from_numpy(feats)).numpy()
    pi = Model(args.out).forward(feats)
    similarity = np.sum(reference * pi, axis=1)
    print(f"Pi model vs training model on {len(items)} recordings: "
          f"similarity {similarity.mean():.4f} average, {similarity.min():.4f} worst (1.0 = identical)")
    if similarity.mean() < 0.98:
        raise SystemExit("the Pi model differs too much from the training model")


if __name__ == "__main__":
    main()
