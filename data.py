def load_datasets(parser, args):

    train_dataset = Dataset(
    split='train',
    source_augmentations=source_augmentations,
    random_chunks=True,
    nb_samples=args.nb_train_samples,
    seq_duration=args.seq_dur,
    **dataset_kwargs
    )

    valid_dataset = Dataset(
        split='valid',
        random_chunks=True,
        seq_duration=args.seq_dur,
        nb_samples=args.nb_valid_samples,
        **dataset_kwargs
    )

    return train_dataset, valid_dataset, args


class Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root,
        split='train',
        target_dir='vocals',
        interferer_dirs=['others','bass', 'drums'],
        ext='.wav',
        nb_samples=1000,
        seq_duration=None,
        random_chunks=False,
        sample_rate=44100,
        source_augmentations=lambda audio: audio,
    ):
        """A dataset of that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.
        Example
        =======
        train/vocals/track11.wav -----------------\
        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/
        train/vocals/track11.wav ---------------------> output
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.ext = ext
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_dir = target_dir
        self.interferer_dirs = interferer_dirs
        self.source_folders = self.interferer_dirs + [self.target_dir]
        self.source_tracks = self.get_tracks()
        self.nb_samples = nb_samples

    def __getitem__(self, index):
        # for validation, get deterministic behavior
        # by using the index as seed
        if self.split == 'valid':
            random.seed(index)

        # For each source draw a random sound and mix them together
        audio_sources = []
        for source in self.source_folders:
            # select a random track for each source
            source_path = random.choice(self.source_tracks[source])
            if self.random_chunks:
                duration = load_info(source_path)['duration']
                start = random.uniform(0, duration - self.seq_duration)
            else:
                start = 0

            audio = load_audio(
                source_path, start=start, dur=self.seq_duration
            )
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)
        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the last element in the list
        y = stems[-1]
        return x, y

    def __len__(self):
        return self.nb_samples

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        source_tracks = {}
        for source_folder in tqdm.tqdm(self.source_folders):
            tracks = []
            source_path = (p / source_folder)
            for source_track_path in source_path.glob('*' + self.ext):
                if self.seq_duration is not None:
                    info = load_info(source_track_path)
                    # get minimum duration of track
                    if info['duration'] > self.seq_duration:
                        tracks.append(source_track_path)
                else:
                    tracks.append(source_track_path)
            source_tracks[source_folder] = tracks
        return source_tracks