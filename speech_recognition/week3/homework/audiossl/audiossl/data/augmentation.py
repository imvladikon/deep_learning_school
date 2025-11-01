import torch
from torch_audiomentations import Compose, Gain, AddColoredNoise, PitchShift, Shift


class ContrastiveAudioAugmentation:
    """
    Audio augmentation pipeline for contrastive learning on AudioMNIST

    Creates two different augmented views of the same audio sample.
    Augmentations are chosen to preserve digit intelligibility while
    adding realistic variations.

    Provides augmentations suitable for AudioMNIST (spoken digits):
        - Gaussian noise
        - Gain (volume) changes
        - Small pitch shifts
        - Small time stretching


    Args:
        sample_rate: Audio sample rate (default: 16000 for AudioMNIST)
        p: Probability of applying each augmentation (default: 0.5)
    """

    def __init__(self, sample_rate: int = 16000, p: float = 0.5):
        self.sample_rate = sample_rate

        self.transform = Compose(
            transforms=[
                # background noise (white/pink noise)
                AddColoredNoise(
                    min_snr_in_db=10.0,  # Signal-to-noise ratio
                    max_snr_in_db=30.0,  # Higher = less noise
                    min_f_decay=-2.0,
                    max_f_decay=2.0,
                    p=p,
                    output_type='tensor',
                ),

                # volume variations
                Gain(
                    min_gain_in_db=-6.0,  # Quieter
                    max_gain_in_db=6.0,   # Louder
                    p=p,
                    output_type='tensor',
                ),

                # small pitch shifts
                PitchShift(
                    min_transpose_semitones=-2.0,  # Slightly lower pitch
                    max_transpose_semitones=2.0,   # Slightly higher pitch
                    sample_rate=sample_rate,
                    p=p,
                    output_type='tensor',
                ),

                # time shift (not sure if stretching is good for digits)
                # Shift(
                #     min_shift=-0.2,  # Shift left
                #     max_shift=0.2,   # Shift right
                #     p=p,
                #     output_type='tensor',
                # ),
            ],
            output_type='tensor',
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # torch-audiomentations expects [batch, channels, time]
        if waveform.ndim == 1:
            # [time] -> [1, 1, time]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            augmented = self.transform(waveform, sample_rate=self.sample_rate)
            return augmented.squeeze(0).squeeze(0)

        elif waveform.ndim == 2:
            # [batch, time] -> [batch, 1, time]
            waveform = waveform.unsqueeze(1)
            augmented = self.transform(waveform, sample_rate=self.sample_rate)
            return augmented.squeeze(1)

        else:
            # [batch, channels, time] - already correct
            return self.transform(waveform, sample_rate=self.sample_rate)


def create_contrastive_views(waveform: torch.Tensor, augment_fn, spec_transform=None):
    """
    Create two augmented views for contrastive learning

    Strategy: Apply different random augmentations to create diverse views

    Args:
        waveform: [batch_size, time] - input waveforms
        augment_fn: Augmentation function
        spec_transform: Optional spectrogram transform

    Returns:
        view1: [batch_size, time] or [batch_size, channels, freq, time]
        view2: [batch_size, time] or [batch_size, channels, freq, time]
    """
    # View 1: Augmented waveform
    view1 = augment_fn(waveform)

    # View 2: Different augmentation of same waveform
    view2 = augment_fn(waveform)

    # If spectrogram transform provided, convert view2
    if spec_transform is not None:
        spectrograms = []
        for wav in view2:
            spec = spec_transform(wav.unsqueeze(0).cpu())
            spectrograms.append(spec)
        view2 = torch.stack(spectrograms).to(waveform.device)

    return view1, view2
