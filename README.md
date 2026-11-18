# voice-rectifier

Custom voice commands on a Raspberry Pi. Record a phrase a few times, then the Pi listens for it.
Works offline, in any language.

## Install

```
sudo apt install python3-numpy alsa-utils
git clone https://github.com/seanghay/voice-rectifier.git
cd voice-rectifier
```

## Usage

```
python3 enroll.py add "lamp on" --mic --count 3
python3 enroll.py add "lamp off" --mic --count 3
python3 listen.py --run "./home.sh {command}"
```

Pick the microphone with `--device plughw:1,0` (see `arecord -l`). Enroll from files with
`enroll.py add "lamp on" a.wav b.wav`. Other commands: `enroll.py list`, `enroll.py remove NAME`.
Use `listen.py -v` to see why something was ignored, and `--threshold` to make it more or less strict.

Commands of two or more words work best.

## How it works

The model maps speech to a 128-dim embedding. Enrolling stores the embeddings of your recordings.
Listening compares each utterance to them and fires the closest command if it is similar enough.

## Accuracy

5 enrolled commands, phrases not seen in training, 1 hour of background talk, thresholds set for
at most 1 false trigger per hour:

| | 1 example | 2 examples | 3 examples |
|---|---|---|---|
| Khmer phrase | 92% | 97% | 98% |
| English two words, unseen speakers | 54% | 67% | 73% |
| Single short word | 36% | 42% | 53% |

## Training

```
pip install -r requirements-train.txt
python prepare.py speech-commands --out data/sc
python train.py --train data/sc/train --val data/sc/val --noise data/sc/noise --out runs/new
python calibrate.py --model runs/new --phrases data/sc/val --background talk/ --write
```

Datasets are folders of `<phrase>/*.wav`, each phrase spoken by several people.
`prepare.py phrases` builds one from a CSV of `path,text_id,speaker`. `train.py` writes a model
folder you can use with `--model runs/new`, or copy over `model/`. After replacing the model,
run `enroll.py rebuild`.

## Credits

Trained on [Speech Commands](https://arxiv.org/abs/1804.03209) (CC BY 4.0) and Khmer read speech.
