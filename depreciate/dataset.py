import os

import numpy as np
import scipy.io.wavfile as wav

import torch
from torch.utils.data import Dataset, DataLoader

#from tool import Transform, create_features, create_inputs
import generate_masking_threshold as generate_mask

class AudioDataset(Dataset):

    def __init__(self, manifest_path, batch_size=16, root_dir='./', window_size=2048):

        # data directories
        self.manifest_path = manifest_path
        self.root_dir = root_dir

        loaded_manifest = np.loadtxt(self.manifest_path, dtype=str, delimiter=",")
        self.audio_pathes = loaded_manifest[0]
        self.original_phrase = loaded_manifest[1]
        self.target_phrase = loaded_manifest[2]

        # data preprocessing
        self.window_size = window_size

        # training parameters
        self.batch_size = batch_size

        # list of data
        self.sample_rates = []
        self.audios = []
        self.audio_lengths = []

        # read and store audio file
        for audio_path in self.audio_pathes:

            sample_rate, audio = wav.read(os.path.join(self.root_dir, audio_path))
            if max(audio) < 1: # set audio to [-32767, 32768] if needed
                audio = audio * 32768

            self.sample_rates.append(sample_rate)
            self.audios.append(audio)
            self.audio_lengths.append(len(audio))

    def __getitem__(self, idx):
        return {'audio': self.audios[idx],
                'length': self.audio_lengths[idx],
                'target': self.target_phrase[idx],
                'sr': self.sample_rates[idx]}

    def __len__(self):
        return len(self.audios)

    def dataloader(self):
        return DataLoader(self, batch_size=self.batch_size,
                                collate_fn=AudioDataset.collate_fn,
                                shuffle=False)

    @staticmethod
    def collate_fn(batch):

        unpacked_batch = [(x['audio'], x['length'], x['target'], x['sr']) for x in batch]
        audios, lengths, targets, sample_rates = zip(*unpacked_batch)

        max_len = max(lengths)
        batch_size = len(batch)

        # create padded array
        padded_audios = np.zeros([batch_size, max_len])
        masks = np.zeros([batch_size, max_len])

        # prepare frequency array for masking
        freq_lengths = (np.array(lengths) // 2 + 1) // 240 * 3
        max_freq_lengths = max(freq_lengths)
        freq_masks = np.zeros([batch_size, max_freq_lengths, 80])
        thresholds = []
        psd_maxes = []

        for idx in range(batch_size):

            padded_audios[idx, :lengths[idx]] = audios[idx].astype(float)
            masks[idx, :lengths[idx]] = 1

            # compute the masking threshold
            threshold, psd_max = generate_mask.generate_th(padded_audios[idx],
                                                           sample_rates[idx])
            # threshold, psd_max = 0, 0
            freq_masks[idx, :freq_lengths[idx], :] = 1
            thresholds.append(threshold)
            psd_maxes.append(psd_max)

        thresholds = np.array(thresholds)
        psd_maxes = np.array(psd_maxes)

        return {'audio': torch.tensor(padded_audios),
                'audio_length': torch.tensor(lengths),
                'audio_mask': torch.tensor(masks),
                'freq_mask': torch.tensor(freq_masks),
                'target': targets,
                'threshold': torch.tensor(thresholds),
                'psd_max': torch.tensor(psd_maxes),
                }
