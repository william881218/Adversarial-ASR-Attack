from argparse import ArgumentParser
from dataset import AudioDataset
from attack import Attack

def main(args):

    # initialize dataset
    audio_dataset = AudioDataset(args.manifest,
                                 batch_size=args.batch_size,
                                 root_dir=args.root_dir)

    audio_dataloader = audio_dataset.dataloader()


    attack = Attack(args.model_path,
                    batch_size=args.batch_size,
                    lr_stage1=args.lr_stage1,
                    lr_stage2=args.lr_stage2,
                    num_iter_stage1=args.num_iter_stage1,
                    num_iter_stage2=args.num_iter_stage2)

    # initialize attack class
    attack.attack_stage1(audio_dataloader)

    for idx, batch in enumerate(audio_dataloader):
        #print(idx, batch)
        for key, value in batch.items():
            try:
                print(key, ':', value.shape)
            except AttributeError:
                print(key, ':', value)

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
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--lr_stage1', default=100, help='learning_rate for stage 1')
    parser.add_argument('--lr_stage2', default=1, help='learning_rate for stage 2')
    parser.add_argument('--num_iter_stage1', default=1000, help='number of iterations in stage 1')
    parser.add_argument('--num_iter_stage2', default=4000, help='number of iterations in stage 2')

    parser.add_argument('--model_path', default='model/librispeech_pretrained_v2.pth')

    '''
    # data processing
    flags.DEFINE_integer('window_size', '2048', 'window size in spectrum analysis')
    flags.DEFINE_integer('max_length_dataset', '223200',
                        'the length of the longest audio in the whole dataset')
    flags.DEFINE_float('initial_bound', '2000', 'initial l infinity norm for adversarial perturbation')

    # training parameters
    flags.DEFINE_string('checkpoint', "./model/ckpt-00908156",
                        'location of checkpoint')
    flags.DEFINE_integer('gpus', '0', 'which gpu to run')
    '''

    return parser.parse_args()

if __name__ == '__main__':

    args = set_parser()
    main(args)
