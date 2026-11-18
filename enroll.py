"""Teach the Pi a command from a few recordings.

  python3 enroll.py add "lamp on" lamp1.wav lamp2.wav lamp3.wav
  python3 enroll.py add "lamp on" --mic --count 3
  python3 enroll.py list
  python3 enroll.py remove "lamp on"
  python3 enroll.py rebuild          # after switching to a new model
"""

import argparse
import os
import re
import time

import numpy as np

from voice_rectifier import Commands, Model, VoiceDetector, is_speech, load_audio, microphone, normalize, speech_seconds, write_wav


def record(model, count, device):
    rate = model.audio.sample_rate
    mic = microphone(rate, device)
    detector = VoiceDetector(rate)
    print("Measuring background noise, stay quiet...")
    detector.calibrate(np.concatenate([next(mic) for _ in range(10)]))
    takes = []
    while len(takes) < count:
        print(f"[{len(takes) + 1}/{count}] Say the command")
        for block in mic:
            utterance = next(detector.feed(block), None)
            if utterance is None:
                continue
            ok, reason = is_speech(utterance, rate)
            if ok and not detector.too_long:
                takes.append(utterance)
                print(f"  got it ({len(utterance) / rate:.1f}s)")
            else:
                print(f"  didn't catch that ({reason or 'too long'}), try again")
            break
    return takes


def check(commands, name, model):
    examples = np.array([e["embedding"] for e in commands.examples(name)])
    print(f"'{name}': {len(examples)} example(s)")
    bar = model.threshold(1, commands.seconds(name))
    for i in range(len(examples) if len(examples) > 1 else 0):
        others = normalize(np.delete(examples, i, axis=0).mean(axis=0))
        if examples[i] @ others < bar:
            print(f"  example {i + 1} sounds different from the others; consider recording it again")
    for other in commands.names:
        if other != name and commands.prototype(name) @ commands.prototype(other) >= bar:
            print(f"  warning: sounds a lot like '{other}', they may get mixed up")


def add(args, model, commands):
    rate = model.audio.sample_rate
    if args.mic:
        waves = record(model, args.count, args.device)
    elif args.files:
        waves = [load_audio(f, model.audio) for f in args.files]
    else:
        raise SystemExit("give .wav files or --mic")
    folder = os.path.join(os.path.dirname(os.path.abspath(args.commands)), "recordings",
                          re.sub(r"[^\w-]+", "_", args.name))
    os.makedirs(folder, exist_ok=True)
    for i, wave in enumerate(waves):
        ok, reason = is_speech(wave, rate)
        if not ok:
            print(f"  skipped example {i + 1}: {reason}")
            continue
        path = os.path.join(folder, f"{time.strftime('%Y%m%d-%H%M%S')}-{i + 1}.wav")
        write_wav(path, wave, rate)
        commands.add(args.name, model.embed(wave), speech_seconds(wave, model.audio),
                     os.path.relpath(path, os.path.dirname(os.path.abspath(args.commands))))
    if args.name not in commands.names:
        raise SystemExit("no usable recordings, nothing saved")
    if args.threshold is not None:
        commands.data["commands"][args.name]["threshold"] = args.threshold
    commands.save()
    check(commands, args.name, model)


def rebuild(args, model, commands):
    base = os.path.dirname(os.path.abspath(args.commands))
    for name in commands.names:
        for example in commands.examples(name):
            wave = load_audio(os.path.join(base, example["audio"]), model.audio)
            example["embedding"] = [round(float(v), 6) for v in model.embed(wave)]
        print(f"rebuilt '{name}'")
    commands.data["model_id"] = model.model_id
    commands.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model")
    parser.add_argument("--commands", default="commands.json")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("add")
    p.add_argument("name")
    p.add_argument("files", nargs="*")
    p.add_argument("--mic", action="store_true")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--device", help="microphone, e.g. plughw:1,0 (see: arecord -l)")
    p.add_argument("--threshold", type=float, help="custom accept level for this command (0-1)")
    sub.add_parser("list")
    p = sub.add_parser("remove")
    p.add_argument("name")
    sub.add_parser("rebuild")
    args = parser.parse_args()

    model = Model(args.model)
    if args.action == "rebuild":
        commands = Commands(args.commands, None)
        rebuild(args, model, commands)
        return
    commands = Commands(args.commands, model.model_id)
    if args.action == "add":
        add(args, model, commands)
    elif args.action == "list":
        for name in commands.names:
            print(f"{name!r:30} {len(commands.examples(name))} example(s), "
                  f"accept level {commands.threshold(name, model):.2f}")
    elif args.action == "remove":
        if not commands.remove(args.name):
            raise SystemExit(f"no command named '{args.name}'")
        commands.save()
        print(f"removed '{args.name}'")


if __name__ == "__main__":
    main()
