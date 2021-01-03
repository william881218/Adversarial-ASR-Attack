import random
import os

from flask import Flask, request, jsonify, render_template, send_file
import logging
from scipy.io import wavfile
import torch
import numpy as np

from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from asr_attack import AsrAttack, ATTACK_PARAMS


# instantiate flask app
app = Flask(__name__)

app.config['ROOT_DIR'] = ROOT_DIR = os.path.dirname(__file__)
app.config['UPLOAD_DIR'] = os.path.join(ROOT_DIR, 'upload') # for uploaded files
app.config['AUDIO_DIR'] = os.path.join(ROOT_DIR, 'output') # for generated files
app.config['OUTPUT_PATH'] = os.path.join(app.config['AUDIO_DIR'], 'perturbed.wav')


@app.route('/', methods=['GET'])
def main_page():
    '''
    Return the main page.
    '''
    return render_template('index.html')


@app.route('/corpus/7.wav', methods=['GET'])
def download_corpus():
    '''
    Responsible for downloading audio corpuses. 
    Currently only 7.wav is available.
    '''
    return send_file(os.path.join(app.config["ROOT_DIR"], "corpus/7.wav"), as_attachment=True)


@app.route('/perturbed_audio', methods=['GET'])
def download_perturbed_audio():
    '''
    Download the perturbed audio.
    Note that every time a perturbed audio is generated,
    the old one on server will be replaced.
    '''
    return send_file(app.config['OUTPUT_PATH'], as_attachment=True)


@app.route("/attack", methods=["POST"])
def attack():
    """Endpoint to do the asr attack
    :return (json): This endpoint returns a json file with the following format:
        {
            "keyword": "down"
        }
    """

    # prepare the path where the uploaded/generated audio are saved
    file_idx = str(random.randint(0, 100000)) + '.wav'
    uploaded_path = os.path.join(app.config['UPLOAD_DIR'], file_idx)
    output_path = os.path.join(app.config['AUDIO_DIR'], file_idx)

    # get file from POST request and save it
    audio_file = request.files["file"]
    audio_file.save(uploaded_path)

    # get target from POST request
    target_phrase = request.form["target"]

    # instantiate keyword spotting service singleton and get prediction
    my_asr_attack = AsrAttack(**ATTACK_PARAMS)
    my_asr_attack.generate_adv_example(input_path=uploaded_path,
                                       target=target_phrase, 
                                       output_path=app.config['OUTPUT_PATH'])
    
    # we don't need the audio file any more - let's delete it!
    os.remove(uploaded_path)

    # send back html page with result
    attack_result = {"output_path": "/perturbed_audio"}
    return render_template('attack_success.html', attack_result=attack_result)


@app.route("/asr", methods=["POST"])
def asr():
    """Endpoint to use asr
    :return (json): This endpoint returns a json file with the following format:
        {
            "transcription": TRANSCRIPTION
        }
    """
    
    # prepare the path where the uploaded/generated audio are saved
    file_idx = str(random.randint(0, 100000)) + '.wav'
    uploaded_path = os.path.join(app.config['UPLOAD_DIR'], file_idx)
    output_path = os.path.join(app.config['AUDIO_DIR'], file_idx)

    # get file from POST request and save it
    audio_file = request.files["file"]
    audio_file.save(uploaded_path)
    logging.info("File saved at {}".format(uploaded_path))

    # select device here
    if torch.cuda.is_available():
        device_type = "gpu"
    else:
        device_type = "cpu"

    asr_model = PyTorchDeepSpeech(pretrained_model="librispeech",
                                   device_type=device_type)

    # load audio
    sample_rate, sound = wavfile.read(uploaded_path)

    if sample_rate != 16000: # check if it has valid sample rate
        transcription = "SAMPLE_RATE_ERROR"
        logging.info("Sample rate error.")
    else: # start prediction
        transcription = asr_model.predict(np.array([sound]), batch_size=1,transcription_output=True)[0]
        logging.info("Finish prediction. Transcription: {}".format(transcription))
    
    # we don't need the audio file any more - let's delete it!
    os.remove(uploaded_path)

    # send back html page with result
    transcription = {"text": transcription}
    return render_template('transcription.html', transcription=transcription)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9527')