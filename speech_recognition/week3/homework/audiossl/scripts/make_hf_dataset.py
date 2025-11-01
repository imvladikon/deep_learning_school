"""
AudioMNIST HuggingFace Dataset Creator

Usage:
    python make_hf_dataset.py

    # custom paths
    python make_hf_dataset.py --wav-dir /path/to/wavs --output-dir /path/to/output

    # split ratios
    python make_hf_dataset.py --train-ratio 0.7 --val-ratio 0.15

    # custom sampling rate
    python make_hf_dataset.py --sampling-rate 48000

    
Note:
    check https://github.com/huggingface/datasets/issues/7834
    if there are some issues with audio decoding (memory allocation errors)
""" # noqa: E501

from pathlib import Path
import re
from tqdm import tqdm
from datasets import Dataset, DatasetDict, disable_caching, Features, Value
from collections import Counter
import random
import argparse

disable_caching()

# Примеры шаблонов: "3_04_23.wav" -> label=3, speaker=04, utt=23
#                   "03_4_023.wav" -> label=03, speaker=4, utt=023
rx = re.compile(
    r"(?P<label>\d+)[_\-](?P<speaker>\d+)[_\-](?P<utt>\d+)\.wav$", re.IGNORECASE
)


def data_generator(wav_dir):
    for p in tqdm(wav_dir.rglob("*.wav"), desc="Collecting audio files"):
        m = rx.search(p.name)
        if not m:
            continue
        label = int(m.group("label"))
        speaker = int(m.group("speaker"))
        utt = int(m.group("utt"))
        yield {
            "audio": str(p),
            "label": label,
            "speaker_id": speaker,
            "utterance_id": utt,
        }


def select_by_speakers(d, spkset):
    mask = [s in spkset for s in d["speaker_id"]]
    return d.select([i for i, m in enumerate(mask) if m])


def main():
    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from AudioMNIST wav files"
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path(__file__).parent / "AudioMNIST" / "data",
        help="Path to directory containing wav files (default: ./AudioMNIST/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "hf_audio_mnist",
        help="Path to save the processed dataset (default: ./hf_audio_mnist)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for speaker split (default: 0)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of speakers for training set (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of speakers for validation set (default: 0.1)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Audio sampling rate in Hz (default: 16000)",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    wav_dir = args.wav_dir

    if not wav_dir.exists():
        raise RuntimeError(f"Directory {wav_dir} does not exist")

    features = Features(
        {
            "audio": Value("string"),
            "label": Value("int64"),
            "speaker_id": Value("int64"),
            "utterance_id": Value("int64"),
        }
    )

    columns = {name: [] for name in features}

    for example in data_generator(wav_dir):
        for key in columns:
            columns[key].append(example[key])

    if not columns["audio"]:
        raise RuntimeError(f"Не нашли ни одного .wav в {wav_dir}")

    ds = Dataset.from_dict(columns, features=features)

    speakers = sorted(set(ds["speaker_id"]))
    random.shuffle(speakers)

    n = len(speakers)
    train_end = int(args.train_ratio * n)
    val_end = train_end + int(args.val_ratio * n)

    train_spk = set(speakers[:train_end])
    val_spk = set(speakers[train_end:val_end])
    test_spk = set(speakers[val_end:])

    ds_train = select_by_speakers(ds, train_spk)
    ds_val = select_by_speakers(ds, val_spk)
    ds_test = select_by_speakers(ds, test_spk)

    from datasets import Audio

    audio_feature = Audio(sampling_rate=args.sampling_rate)
    ds = DatasetDict(
        {
            "train": ds_train.cast_column("audio", audio_feature),
            "validation": ds_val.cast_column("audio", audio_feature),
            "test": ds_test.cast_column("audio", audio_feature),
        }
    )

    for split in ds:
        c = Counter(ds[split]["label"])
        print(split, len(ds[split]), c)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(
        f"Saved to {args.output_dir}. Cast with datasets.Audio(decode=True) after loading when you need arrays."
    )


if __name__ == "__main__":
    main()
