from flask import Flask, render_template, request, Response
from speed_stream import generate_stream
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_path = ""  # global path for uploaded video

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    return "Uploaded", 200

@app.route('/video_feed')
def video_feed():
    global video_path
    return Response(generate_stream(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
