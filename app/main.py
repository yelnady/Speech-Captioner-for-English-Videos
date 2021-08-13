from flask import Flask, render_template, request

import numpy as np
import os

import youtube_dl
from time import time
from datetime import timedelta
import moviepy.editor as mp
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from threading import Thread, Event
import tensorflow as tf
tf.keras.backend.clear_session()

processor = Wav2Vec2Processor.from_pretrained("saved_model")
model = Wav2Vec2ForCTC.from_pretrained("saved_model")

# port = 12300

threads = [Thread()]
def project_id():
    import json
    info = json.load(
        open(os.path.join(os.environ['HOME'], ".smc", "info.json"), 'r'))
    return info['project_id']


#base_url = "/%s/port/%s/" % (project_id(), port)
#static_url = "/%s/port/%s/static" % (project_id(), port)
app = Flask(__name__)

def youtube_download(youtubeURL):

    name_of_youtube_output_mp4 = 'video_original.mp4'
    info = None
    ydl_opts = {
        'outtmpl': 'static/' + name_of_youtube_output_mp4,
        'continuedl': False,
        'format':'best'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtubeURL, download=True)
    view_count = info['view_count']
    duration = info['duration']
    video_title = info['title']   
    print(video_title)
    
def video_to_audio(name_of_youtube_output_mp4):
    name_of_audio_output = 'audio_generated.wav'

    clip = mp.VideoFileClip(name_of_youtube_output_mp4)
   
    clip.audio.write_audiofile(
        'static/'+ name_of_audio_output, 16_000)
    
    audio_full_path ='static/' + name_of_audio_output
    return audio_full_path 

def wav2vec(name_of_audio_output):

    lines = ['WEBVTT', '']
    audio_input, sample_rate = sf.read(name_of_audio_output)
    audio_input = audio_input.sum(axis=1) / 2

    for idx, start_frame in enumerate(
            range(0, audio_input.shape[0], 16000 * 10)):
        end_frame = start_frame + 16000 * 10
        # pad input values and return pt tensor
        input_values = processor(
            audio_input[start_frame:end_frame],
            sampling_rate=sample_rate,
            return_tensors="pt").input_values

        # INFERENCE
        # retrieve logits & take argmax
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # transcribe
        transcription = processor.decode(predicted_ids[0])
        lines.append(
            str(timedelta(seconds=idx * 10)) + '.000 --> ' +
            str(timedelta(seconds=(idx + 1) * 10)) + '.000')

        lines.append(transcription.lower())

        lines.append('')

    with open('static/'+'subtitles.vtt', 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))

    return 1


@app.route('/')
def home():
    name = "Audio Captioner - Universal"
    return render_template('home.html', name=name)

@app.route('/youtube-url', methods=['POST', 'GET'])
def youtube_url():
    global threads
    print("Function youtube_url has been called\n\n")
    youtube_url = request.args.get('youtube_url')
    if os.path.exists("static/video_original.mp4"):
        os.remove("static/video_original.mp4")
    if os.path.exists("static/subtitles.vtt"):
        os.remove("static/subtitles.vtt")
    # I was calling the thread directly from here --> All I am doing now, is to call the thread from within the socket
    x = Thread(target=long_runnning_job, args=(youtube_url,True, None))
    x.start()
    threads[0] = x
    print("Thread is called and running in the background\n\n")
    return 'I am a title'


@app.route('/video-selected', methods=['POST', 'GET'])
def video_selected():
    global threads
    print("Function video_selected has been called\n\n")
    # We should this time get a file from the frontend, not youtube url
    file_object =request.files.get('file')
    if os.path.exists("static/video_original.mp4"):
        os.remove("static/video_original.mp4")
    if os.path.exists("static/subtitles.vtt"):
        os.remove("static/subtitles.vtt")
    file_object.save('static/video_original.mp4')
    
    x = Thread(target=long_runnning_job, args=(None,False,file_object))
    x.start()
    threads[0] = x
    print("Thread is called and running in the background\n\n")
    return 'I am a title'


def long_runnning_job(youtube_url,is_it_youtube,file_object):
    time_all = time()

    print("Started Thread\n\n")
    if is_it_youtube:
        youtube_download(youtube_url)
        
    audio_full_path = video_to_audio('static/video_original.mp4')
    print("Started Generating Subtitles")
    wav2vec(audio_full_path)
    print('TRANSCRIPT IS READY!!')
    print("Total Time (in minutes) is {}\n\n".format(timedelta(seconds=(time() - time_all))))
    return 'I am a title'

@app.route('/thread-check', methods=['POST', 'GET'])
def thread_check():
    global threads
    return "1" if threads[0].is_alive() else "0"
    
if __name__ == "__main__":
    # you will need to change code.ai-camp.org to other urls if you are not running on the coding center.
    print(
        "Try to open\n\n    https://cocalcg12.ai-camp.org" + base_url + '\n\n')
    # We don't need to call app.run(), we just call socketio to run everything since it's a flask wrapper.
    # We need also to have .js and .map file in static, and reference them from the client or html side
    app.run( host='0.0.0.0', port=port, debug=True)
    import sys
    sys.exit(0)
