from flask import Flask, request, Response
from flask_cors import CORS
from speed_stream import generate_stream
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_path = ""

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    return "Uploaded", 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)