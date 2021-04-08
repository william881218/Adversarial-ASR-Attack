from argparse import ArgumentParser
import os

from asr_attack import AsrAttack, ATTACK_PARAMS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):

    my_asr_attack = AsrAttack(**ATTACK_PARAMS, debug=True, gpus="0")
    my_asr_attack.generate_adv_example(input_path=args.input_path, 
                                       target=args.target, 
                                       output_path=args.output_path)


def set_parser():

    parser = ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str, 
        default="/home/b07902027/Audio-Captcha/audio_dataset/" + 
        "en-US-Standard-H/2/en-US-Standard-H_2_0.0.wav"
    )
    parser.add_argument("-t", "--target", type=str, default="THREE")
    parser.add_argument("-o", "--output_path", type=str, default="/home/b07902027/adversarial_asr_pytorch/adv_example/out.wav")

    return parser.parse_args()


if __name__ == '__main__':
    args = set_parser()
    main(args)
