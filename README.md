# voice-rectifier

Voice commands for a Raspberry Pi, in any language. You record a command 1 to 3 times, and the Pi
listens for it from then on. No training, no internet, no speech-to-text.

```
python3 enroll.py add "lamp on" --mic
python3 listen.py --run "./lamp.sh {command}"
```

## How it works

A small model turns a bit of speech into a list of 128 numbers, a voice print. Recordings of the
same phrase get similar voice prints, no matter who says it or what language it is in. When you
add a command, the Pi stores the voice prints of your recordings. When it hears speech, it compares
it to every command and acts on the closest one, but only if it is close enough. Anything else,
like TV or people talking, is ignored.

The model was trained once, on thousands of phrases in English and Khmer read by many people.
It never needs training again: each home just records its own commands.

## Setup

On a Raspberry Pi 4 or 5 with a USB microphone:

```
sudo apt install python3-numpy alsa-utils
git clone https://github.com/seanghay/voice-rectifier.git
cd voice-rectifier
arecord -l      # find your microphone, e.g. card 1 -> use --device plughw:1,0
```

NumPy is the only Python package needed.

## Add commands

```
python3 enroll.py add "lamp on" --mic --count 3 --device plughw:1,0
python3 enroll.py add "lamp off" --mic --count 3 --device plughw:1,0
```

Stay quiet for a second at the start while it measures the room, then say the command each time
you are asked. You can also use recordings: `python3 enroll.py add "lamp on" a.wav b.wav c.wav`.

```
python3 enroll.py list                  # show your commands
python3 enroll.py add "lamp on" d.wav   # add another recording to a command
python3 enroll.py remove "lamp on"
```

Commands are saved in `commands.json` and the recordings in `recordings/`.

## Listen

```
python3 listen.py --device plughw:1,0
python3 listen.py --device plughw:1,0 --run "./home.sh {command}"
```

`--run` starts a program each time a command is heard, with `{command}` replaced by its name.
Add `-v` to see everything it hears and why it was ignored.

## Tips

- **Use two or more words.** "lamp on" works much better than "on". Single short words are easily
  confused with bits of normal talk.
- **Record 3 times**, in the room and at the distance you will really use.
- **Pick commands that sound different.** `enroll.py` warns you when two commands sound alike.
- **Speak the command on its own**, with a short pause before and after.
- If commands are missed, lower the accept level: `python3 listen.py --threshold 0.45`.
  If it reacts to things you did not say, raise it.

## How well it works

Tested with 5 commands on phrases the model never heard during training, while an hour of
non-stop talk played in the background. The accept level is set so that at most 1 thing per hour
is mistaken for a command.

| Command | 1 recording | 2 recordings | 3 recordings |
|---|---|---|---|
| Khmer phrase, 0.8 s or longer | 92% caught | 97% caught | 98% caught |
| English two-word phrase, new voices | 54% caught | 67% caught | 73% caught |
| Single short word | 36% caught | 42% caught | 53% caught |

The Khmer test phrases were new, but some of the voices were heard during training, so that row is
on the hopeful side. The English rows use voices the model never heard. The background talk was Khmer
news reading; music, English TV and children were not tested.

## Train your own model

You only need this to improve the model. Use a computer with a GPU.

```
pip install -r requirements-train.txt
python -m training.prepare_speech_commands --out data/sc
python -m training.prepare_phrases --list readings.csv --out data/phrases --phrases 3000
python -m training.train --train data/sc/train data/phrases/train --val data/sc/val data/phrases/val \
                         --noise data/sc/noise --out runs/new --audiomentations
python -m training.export runs/new/best.pt --check data/sc/val --out model
python -m training.calibrate --phrases data/sc/val data/phrases/val --background talk/ --write
```

Training data is folders of recordings, one folder per phrase, each phrase said by several
people. `prepare_phrases` builds this from any set of readings where several people read the
same text: give it a CSV with the columns `path,text_id,speaker`. During training, every recording
is randomly changed to sound like other microphones, rooms, noise and volumes. `calibrate` sets
how strict the model is by playing long recordings of ordinary talk (`--background`).

After installing a new model, run `python3 enroll.py rebuild` to update your commands.

## Files

```
enroll.py, listen.py     what you run on the Pi
voice_rectifier/         audio, voice detection, the model, your commands (NumPy only)
model/                   the trained model (model.npz, model.json) and its training checkpoint
training/                everything for training a new model (needs PyTorch)
```

## Credits

Trained with [Google Speech Commands](https://arxiv.org/abs/1804.03209) (CC BY 4.0) and a
Khmer read-speech dataset.
