from cProfile import label
import os
from werkzeug.utils import secure_filename
from urllib.request import Request
from flask import Flask, render_template, Response, request, redirect, flash
from Myfunctions import *
import urllib
import secrets
import cv2

secret = secrets.token_urlsafe(32)

app = Flask(__name__)
app.secret_key = secret
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    """my main page"""
    return render_template('index.html')


@app.route('/ImageStream', methods = ['POST', 'GET'])
def ImageStream():
    """the live page"""
    return render_template('RealtimeImage.html')

@app.route('/video_feed', methods = ['POST', 'GET'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/takeimage', methods = ['POST', 'GET'])
def takeimage():
    """ Captures Images from WebCam, saves them and check face mask detection analysis """

    _, frame = camera.read()
    filename = "capture.png"
    cv2.imwrite(f"static\{filename}", frame)
    camera.release()

    results = image_preprocessing(filename)
    if results is None:
        return render_template('Error.html')
    else:

        img_preds = results[0]
        frame = results[1]
        faces_detected = results[2]

        results2 = predictions_results(img_preds, frame, faces_detected, filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        return render_template('PictureResult.html', user_image=full_filename,
                               number_of_face="Number of faces detected: {}".format(results2[0]),
                               no_mask_face="No face mask count: {}".format(results2[1]),
                               correct_mask_face="Correct face mask count: {}".format(results2[2]),
                               incorrect_mask_face="Incorrect face mask count: {}".format(results2[3]))


@app.route('/UploadImage', methods=['GET','POST'])
def UploadImage():
    """the upload image page"""
    return render_template('UploadPicture.html')

if __name__ == '__main__':
    app.run(debug=True)
