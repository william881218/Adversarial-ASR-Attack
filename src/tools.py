import torch

def generate_spectrogram(audio):
    '''
    Args:
        audio (tensor): (B x T)
    Returns:
        mag (tensor): (B x T)
    '''

    n_fft = 320
    hop_length = 160

    # STFT
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, normalized=True)
    print('spec shape: ', spec.shape)
    real = spec[:, :, :, 0] # real part
    imag = spec[:, :, :, 1] # imagine part
    mag = torch.abs(torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))) # magnitude
    mag = torch.log1p(mag)

    return mag
