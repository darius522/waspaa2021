import torch
import torch.utils.data
import argparse
import random
import musdb
import torch
import tqdm

from utils import load_audio, load_info

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(parser, args, train=None, valid=None):

    args = parser.parse_args()

    dataset_kwargs = {
        'root': args.root,
        'seq_duration': args.seq_dur
    }

    train_dataset = Dataset(
        split='train',
        random_chunks=True,
        paths=train,
        **dataset_kwargs
    )
    valid_dataset = Dataset(
        split='valid',
        paths=valid,
        **dataset_kwargs
    )

    return train_dataset, valid_dataset, args


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        input_file='mixture.wav',
        output_file='vocals.wav',
        seq_duration=None,
        random_chunks=False,
        sample_rate=44100,
        paths=None
    ):

        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        self.paths = paths
        # set the input and output files (accept glob)
        self.tuple_paths = list(self._get_paths())
        if not self.tuple_paths:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):
        path = self.tuple_paths[index]

        if self.random_chunks:
            info = load_info(path)
            duration = info['samples']
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0

        audio = load_audio(path, start=start, dur=self.seq_duration)

        # return torch tensors
        return audio, audio

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads track"""

        random.shuffle(self.paths)
        for path in tqdm.tqdm(self.paths):
            if path and self.seq_duration is not None:
                info = load_info(path)
                duration = info['samples']
                if info['samples'] > self.seq_duration:
                    yield path
            else:
                yield path