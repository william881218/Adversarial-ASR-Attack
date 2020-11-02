import torch
from tqdm import tqdm
from deepspeech_pytorch.utils import load_model, load_decoder
from tools import generate_spectrogram
import json

from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model

class Attack:
    def __init__(self, model_path, gpus=None,
                                   batch_size=1,
                                   lr_stage1=100,
                                   lr_stage2=0.1,
                                   num_iter_stage1=1000,
                                   num_iter_stage2=4000,
                                   labels_path='labels.json'):

        # handle attacked model
        self.device = torch.device("cuda" if gpus is None else gpus)
        self.model = load_model(device=self.device,
                                model_path=model_path,
                                use_half=False)
        self.model.eval()

        # handle training parameters
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.batch_size = batch_size
        self.lr_stage1 = lr_stage1

        with open(labels_path) as label_file:
            label = json.load(label_file)
            self.text2label = { x : idx for idx, x in enumerate(label)}
            self.label2text = { idx : x for idx, x in enumerate(label)}

        self.ctc_loss = torch.nn.CTCLoss(blank=len(self.text2label) - 1)


    def attack_stage1(self, data_loader):
        '''
        Original attack

        In each batch of data_loader:
            audio: the original wavform (batch_size, max_len)
            audio_len: length of each waveform (batch_size)
            audio_mask: 0/1 mask to maskout padding of waveform (batch_size, max_len)
            freq_mask: 0/1 mask to maskout mfcc (batch_size, max_freq, 80)
            target: target transcribe (batch_size)
            threshold: a numpy array of the masking threshold, each of size (?, 1025)
            psd_max: a numpy array of the psd_max of the original audio (batch_size)
        '''

        print('attack stage 1...')

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

            audios = batch['audio'].to(self.device)
            audio_lens = batch['audio_length']
            audio_masks = batch['audio_mask']
            targets = batch['target']

            print('input :', audios.shape, audio_lens)

            spec = generate_spectrogram(audios)
            spec = spec.unsqueeze(0).float()
            length = torch.tensor([spec.shape[-1]])
            print('in shape ', spec.shape)

            out, output_sizes = self.model(spec, length)
            prediction = torch.argmax(out, dim=2)
            prediction = ''.join([self.label2text[x] for x in prediction[0].tolist()])
            print('out shape', out.shape)
            print(output_sizes)
            print('prediction: ', prediction)
            exit()

            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()

                # compute gradient
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.optim.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

        # init model
        for batch in data_loader:

            audio, audio_length, audio_mask, freq_mask, target, threshold, psd_max = batch.values()
            print('audio:', audio.shape)
            print('audio_len:' ,audio_length)
            print('audio_mask: ', audio_mask.shape)
            print('freq_mask: ', freq_mask)
            print('target: ', target)
            print('threshold: ', threshold.shape)
            print('psd_max:', psd_max.shape)


            exit()

        # show the initial predictions
        for i in range(self.batch_size):
            print("example: {}, loss: {}".format(num_loop * self.batch_size + i, losses[i]))
            print("pred:{}".format(predictions['topk_decoded'][i, 0]))
            print("targ:{}".format(trans[i].lower()))
            print("true: {}".format(data[1, i].lower()))



        for i in range(MAX):
            now = time.time()


            clock += time.time() - now

        return final_deltas
