import json
import os
import time

import numpy as np

from .model import normalize


class Commands:
    """The user's commands, saved in a JSON file. Each command keeps the voice print of every example."""

    def __init__(self, path, model_id):
        self.path = path
        self.data = {"model_id": model_id, "commands": {}}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
            if self.data["commands"] and self.data["model_id"] != model_id:
                raise SystemExit(f"{path} was made with a different model. Run: python3 enroll.py rebuild")
            self.data["model_id"] = model_id

    @property
    def names(self):
        return sorted(n for n, c in self.data["commands"].items() if c["examples"])

    def examples(self, name):
        return self.data["commands"][name]["examples"]

    def add(self, name, embedding, seconds, audio_file=None):
        cmd = self.data["commands"].setdefault(name, {"examples": [], "threshold": None})
        cmd["examples"].append({"embedding": [round(float(v), 6) for v in embedding],
                                "seconds": round(float(seconds), 3), "audio": audio_file,
                                "added": time.strftime("%Y-%m-%d %H:%M:%S")})

    def remove(self, name):
        return self.data["commands"].pop(name, None) is not None

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def prototype(self, name):
        return normalize(np.mean([e["embedding"] for e in self.examples(name)], axis=0))

    def seconds(self, name):
        return float(np.mean([e["seconds"] for e in self.examples(name)]))

    def threshold(self, name, model):
        custom = self.data["commands"][name].get("threshold")
        return custom if custom is not None else model.threshold(len(self.examples(name)), self.seconds(name))

    def match(self, embedding, seconds, model, threshold=None, margin=0.03, length_tolerance=1.6):
        """Find the best command for a voice print. Returns (command or None, details)."""
        names = self.names
        if not names:
            return None, {"reason": "no commands"}
        scores = np.array([self.prototype(n) @ embedding for n in names])
        order = np.argsort(-scores)
        best = names[order[0]]
        score = float(scores[order[0]])
        runner_up = float(scores[order[1]]) if len(names) > 1 else -1.0
        bar = threshold if threshold is not None else self.threshold(best, model)
        lengths = [e["seconds"] for e in self.examples(best)]
        shortest, longest = min(lengths) / length_tolerance - 0.1, max(lengths) * length_tolerance + 0.1
        info = {"best": best, "score": round(score, 3), "threshold": round(bar, 3)}
        if score < bar:
            info["reason"] = "not similar enough"
        elif score - runner_up < margin:
            info["reason"] = f"too close to '{names[order[1]]}'"
        elif length_tolerance and not shortest <= seconds <= longest:
            info["reason"] = f"wrong length ({seconds:.1f}s)"
        else:
            return best, info
        return None, info
