from flask import Flask, render_template, Response, jsonify
from controller import VideoCamera
import cv2
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

video_stream_vgg = VideoCamera("vgg")
video_stream_resnet = VideoCamera("wide_resnet")

@app.route('/')
def index():
    return render_template('./index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_vgg')
def video_feed_vgg():
    return Response(gen(video_stream_vgg),
                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vgg_page')
def vgg_page():
    return render_template('./vgg_page.html')

@app.route('/resnet_page')
def resnet_page():
    return render_template('./resnet_page.html')

@app.route('/video_feed_resnet')
def video_feed_resnet():
    return Response(gen(video_stream_resnet),
                mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False,port="5000")