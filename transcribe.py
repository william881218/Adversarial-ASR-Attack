from argparse import ArgumentParser
import os

from scipy.io import wavfile
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
import torch
import numpy as np


def main(args):

    # select device here
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device_type = "gpu"
    else:
        device_type = "cpu"

    deepspeech = PyTorchDeepSpeech(pretrained_model=args.model,
                                   device_type=device_type)

    # load audio
    sample_rate, sound = wavfile.read(args.input)
    assert sample_rate == 16000, "This module only supports audio with sample rate of 16000 currently."

    # start prediction
    transcription = deepspeech.predict(np.array([sound]), batch_size=1,transcription_output=True)

    print("output:", transcription)
    

def set_parser():

    parser = ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default="/home/b07902027/adversarial_asr_pytorch/adv_example/out.wav")
    parser.add_argument("-m", "--model", type=str, default="librispeech")
    parser.add_argument('-g', "--gpus", type=str, default="1")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = set_parser()
    main(args)
