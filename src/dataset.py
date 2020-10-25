from argparse import ArgumentParser
import os
import time

import numpy as np
import scipy.io.wavfile as wav

from torch.utils.data import Dataset, DataLoader

#from tool import Transform, create_features, create_inputs
#import generate_masking_threshold as generate_mask

class AudioDataset(Dataset):

    def __init__(self, manifest_path, batch_size=16, root_dir='./'):

        # data directories
        self.manifest_path = manifest_path
        self.root_dir = root_dir

        loaded_manifest = np.loadtxt(self.manifest_path, dtype=str, delimiter=",")
        self.audio_pathes = loaded_manifest[0]
        self.original_phrase = loaded_manifest[1]
        self.target_phrase = loaded_manifest[2]

        # training parameters
        self.batch_size = batch_size

        # list of data
        self.audios = []
        self.audio_lengths = []

        # read and store audio file
        for audio_path in self.audio_pathes:

            sample_rate, audio = wav.read(os.path.join(self.root_dir, audio_path))
            if max(audio) < 1: # set audio to [-32767, 32768] if needed
                audio = audio * 32768

            self.audios.append(audio)
            self.audio_lengths.append(len(audio))

    def __getitem__(self, idx):
        return {'audio': self.audios[idx],
                'length': self.audio_lengths[idx],
                'target': self.target_phrase[idx]}

    def __len__(self):
        return len(self.audios)

    def dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, collate_fn=AudioDataset.collate_fn)

    @staticmethod
    def collate_fn(batch):

        unpacked_batch = [(x['audio'], x['length'], x['target']) for x in batch]
        audios, lengths, targets = zip(*unpacked_batch)

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
            #threshold, psd_max = generate_mask.generate_th(audios_np[i], sample_rate_np, FLAGS.window_size)
            threshold, psd_max = 0, 0
            freq_masks[idx, :freq_lengths[idx], :] = 1
            thresholds.append(threshold)
            psd_maxes.append(psd_max)

        thresholds = np.array(thresholds)
        psd_maxes = np.array(psd_maxes)
        freq_lengths = (np.array(lengths) // 2 + 1) // 240 * 3 # ?

        return {'audio': padded_audios,
                'audio_length': lengths,
                'audio_mask': masks,
                'freq_mask': freq_masks,
                'target': targets,
                'threshold': thresholds,
                'psd_max': psd_maxes,
                }

def main(args):

    audio_dataset = AudioDataset(args.manifest,
                                 batch_size=args.batch_size,
                                 root_dir=args.root_dir
                                 )

    audio_dataloader = audio_dataset.dataloader()

    for idx, batch in enumerate(audio_dataloader):
        print(idx, batch)
        exit()


def set_parser():
    parser = ArgumentParser()

    # data directories
    parser.add_argument('--manifest', type=str, default='./read_data.txt')
    parser.add_argument('--root_dir', type=str, default='./',
                                       help='directory of dataset')

    # data processing
    # training parameters
    parser.add_argument('--gpus', type=int, default=0, help='which gpu to run')
    parser.add_argument('--batch_size', type=int, default=4)

    '''
    # data processing
    flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
    flags.DEFINE_integer('max_length_dataset', '223200',
                        'the length of the longest audio in the whole dataset')
    flags.DEFINE_float('initial_bound', '2000', 'initial l infinity norm for adversarial perturbation')

    # training parameters
    flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                        'location of checkpoint')
    flags.DEFINE_integer('batch_size', '5', 'batch size')
    flags.DEFINE_float('lr_stage1', '100', 'learning_rate for stage 1')
    flags.DEFINE_float('lr_stage2', '1', 'learning_rate for stage 2')
    flags.DEFINE_integer('num_iter_stage1', '1000', 'number of iterations in stage 1')
    flags.DEFINE_integer('num_iter_stage2', '4000', 'number of iterations in stage 2')
    flags.DEFINE_integer('gpus', '0', 'which gpu to run')
    '''

    return parser.parse_args()

if __name__ == '__main__':

    args = set_parser()
    main(args)
