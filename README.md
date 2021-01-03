# Adversarial ASR Attack

A simple ML application with web interface using PyTorch and flask. Will be dockerized soon.

## Description

Generate adversarial audio example ([paper link](https://arxiv.org/pdf/1903.10346.pdf))

- Generate adversarial example:
    - Input: original audio / target sentence
    - Output: an audio which sounds the same but will be transcribed into target phrase.
- Automatic Speech Recognition
    - Input: An audio file.
    - Output: transcribed text.
    - It can be used to justify whether the adversarial attack is success.
- Example corpus: provided for user testing.
- Note that the length of target sentence shouldn't outscale the length of transcription of original audio.
- Demo
<img src='README/demo.png'>

- Note that the art_toolbox used in the code is from [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox), but several modification, including bug fixing have been made in my version of art_toolbox. Therefore, the art_toolbox forked in my repo is needed. I'll update the instruction for installation soon.

## Getting Started

### Dependency
- Will be uploaded soon.

### Installing & Excecuting

* The docker image of this application will be uploaded soon.