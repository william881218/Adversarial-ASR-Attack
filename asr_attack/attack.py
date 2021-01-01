import sys
import pathlib
import os

import numpy as np
import librosa
from scipy.io import wavfile

import torch
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch


class AsrAttack():
    '''
    This class controls all the configuration and parameters, 
    including parameters for attack and inference.

    The attack used here is from `Trusted-AI/adversarial-robustness-toolbox`.
    Check their github page for more information.

    TODO: Use modified version of the attack module written specifically for audio captcha.
    '''

    SAMPLE_RATE = 16000

    def __init__(self, pretrained_model="librispeech", 
                       gpus="0",
                       debug=False, 
                       **attack_kwargs):
        '''
        Create a class `.AsrAttack` instance.

        Args:
            pretrained_model (str) : The choice of target model. Currently this attack supports 
                                     3 different pretrained models consisting of `an4`, `librispeech`
                                     and `tedlium`, representing which dataset the model was trained with.
            gpus (str) : assign specific gpu to use. Default is "0". 
                         If gpu is unavailable, use cpu instead.
            debug (bool) : whether to print the debug message
            attack_kwargs (dict) : arguments for attack parameters. Read the documentation below.

            Args for `attack_kwargs`:
                estimator (PyTorchDeepSpeech) : A trained estimator.
                initial_eps (float) : Initial maximum perturbation that the attacker can introduce.
                max_iter_1st_stage (int): The maximum number of iterations applied for the first 
                                          stage of the optimization of the attack.
                max_iter_2nd_stage (int): The maximum number of iterations applied for the second 
                                          stage of the optimization of the attack.
                learning_rate_1st_stage (float) : The initial learning rate applied for the first 
                                                  stage of the optimization of the attack.
                learning_rate_2nd_stage (float) : The initial learning rate applied for the second 
                                                  stage of the optimization of the attack.
                optimizer_1st_stage: The optimizer applied for the first stage of the optimization 
                                     of the attack. If `None` attack will use `torch.optim.SGD`.
                optimizer_2nd_stage: The optimizer applied for the second stage of the optimization 
                                     of the attack. If `None` attack will use `torch.optim.SGD`.
                global_max_length (int) : The length of the longest audio signal allowed by this attack.
                initial_rescale (float) : Initial rescale coefficient to speedup the decrease of the 
                                          perturbation size during the first stage of the optimization of the attack.
                rescale_factor (float) : The factor to adjust the rescale coefficient during the first 
                                         stage of the optimization of the attack.
                num_iter_adjust_rescale (int) : Number of iterations to adjust the rescale coefficient.
                initial_alpha (float) : The initial value of the alpha coefficient used in the second 
                                        stage of the optimization of the attack.
                increase_factor_alpha (float) : The factor to increase the alpha coefficient used in the second 
                                                stage of the optimization of the attack.
                num_iter_increase_alpha (int) : Number of iterations to increase alpha.
                decrease_factor_alpha (float) : The factor to decrease the alpha coefficient used in the second stage of the
                                                optimization of the attack.
                num_iter_decrease_alpha (int) : Number of iterations to decrease alpha.
                batch_size (int) : Size of the batch on which adversarial samples are generated.
                use_amp (bool) : Whether to use the automatic mixed precision tool to enable mixed precision training or
                                 gradient computation, e.g. with loss gradient computation. When set to True, this option is
                                 only triggered if there are GPUs available.
                opt_level (str) : Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                                  values are `O0`, `O1`, `O2`, and `O3`.
        '''

        self.pretrained_model = pretrained_model
        self.gpus = gpus
        self.debug = debug
        self.attack_kwargs = attack_kwargs

        # set gpu device here
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            self.device_type = "gpu"
        else:
            self.device_type = "cpu"

        # TODO : Set up optimizer in `attack_kwargs`

        # initialize target asr model
        self.asr_model = PyTorchDeepSpeech(pretrained_model=self.pretrained_model,
                                           device_type=self.device_type)

        # attack!
        self.asr_attack = ImperceptibleASRPyTorch(estimator=self.asr_model, **self.attack_kwargs)

    
    def load_audio(self, path):
        '''
        It's the same loader used by deepspeech-pytorch
        '''
        sound, _ = librosa.load(path, sr=AsrAttack.SAMPLE_RATE)
        if len(sound.shape) > 1:
            sound = sound.mean(axis=1)  # multiple channels, average

        return sound


    def save_audio(self, path, audio):
        '''
        Save audio file. Will be rescaled in 16-bits integer.
        '''

        wavfile.write(path, AsrAttack.SAMPLE_RATE, audio)


    def generate_adv_example(self, input_path, target, output_path):
        '''
        Generate adversarial example.

        Args:
            input_path (str) : the path of audio being attacked.
            target (str) : target output in capital letter. Ex: "OPEN THE DOOR".
            output_path (str) : the path where targeted audio is stored.
        '''
        
        audio = self.load_audio(input_path)
        prediction = self.asr_model.predict(np.array([audio]), batch_size=1, transcription_output=True)
        if self.debug:
            print('input path:', input_path)
            print('original prediction:', prediction)
            print('target:', target)

        # start generating adv example
        adv_audio, first_step_audio = self.asr_attack.generate(np.array([audio]), np.array([target]), batch_size=1)

        # check the transcription of targeted audio
        adv_transcriptions = self.asr_model.predict(adv_audio, batch_size=1, transcription_output=True)
        print("Groundtruth transcriptions: ", prediction)
        print("Target      transcriptions: ", target)
        print("Adversarial transcriptions: ", adv_transcriptions)

        # save adv audio
        self.save_audio(output_path, adv_audio[0])
        if self.debug:
            print('Generated audio stored at:', output_path)