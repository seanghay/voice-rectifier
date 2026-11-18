"""Listen for your commands.

  python3 listen.py                                  # microphone, runs forever
  python3 listen.py --run "./action.sh {command}"    # run a program when a command is heard
  python3 listen.py recording.wav                    # check a recording instead of the mic
"""

import argparse
import shlex
import subprocess
import time

import numpy as np

from voice_rectifier import Commands, Model, VoiceDetector, is_speech, load_audio, microphone, speech_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", nargs="?", help="a recording to check instead of the microphone")
    parser.add_argument("--model", default="model")
    parser.add_argument("--commands", default="commands.json")
    parser.add_argument("--device", help="microphone, e.g. plughw:1,0 (see: arecord -l)")
    parser.add_argument("--run", help="program to run on a command; {command} is replaced by its name")
    parser.add_argument("--threshold", type=float, help="accept level for all commands (0-1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="also show what was ignored and why")
    args = parser.parse_args()

    model = Model(args.model)
    commands = Commands(args.commands, model.model_id)
    if not commands.names:
        raise SystemExit("no commands yet; add some with enroll.py")
    rate = model.audio.sample_rate
    detector = VoiceDetector(rate)

    def hear(utterance, when):
        if detector.too_long:
            reason = "too long"
        else:
            ok, reason = is_speech(utterance, rate)
        if reason:
            if args.verbose:
                print(f"[{when}] ignored: {reason}")
            return
        command, info = commands.match(model.embed(utterance), speech_seconds(utterance, model.audio),
                                       model, args.threshold)
        if command:
            print(f"[{when}] {command}  (score {info['score']:.2f})", flush=True)
            if args.run:
                subprocess.Popen([part.replace("{command}", command) for part in shlex.split(args.run)])
        elif args.verbose:
            print(f"[{when}] unknown: {info['reason']} (closest '{info.get('best')}' "
                  f"{info.get('score', 0):.2f}, needs {info.get('threshold', 0):.2f})")

    if args.wav:
        x = load_audio(args.wav, model.audio)
        step = rate // 10
        for i in range(0, len(x), step):
            for utterance in detector.feed(x[i:i + step]):
                hear(utterance, f"{i / rate:.1f}s")
        for utterance in detector.flush():
            hear(utterance, "end")
        return

    mic = microphone(rate, args.device)
    print("Measuring background noise, stay quiet...")
    detector.calibrate(np.concatenate([next(mic) for _ in range(10)]))
    print(f"Listening for: {', '.join(commands.names)}  (Ctrl+C to stop)")
    try:
        for block in mic:
            for utterance in detector.feed(block):
                hear(utterance, time.strftime("%H:%M:%S"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
