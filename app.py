import os
import requests
from flask import Flask, render_template, request, send_file
from faster_whisper import WhisperModel
import uuid

app = Flask(__name__)

# CPU optimized Whisper model
model = WhisperModel("small", compute_type="int8")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["file"]
    
    unique_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_FOLDER, unique_id + "_" + file.filename)
    file.save(audio_path)

    segments, info = model.transcribe(audio_path)

    srt_path = os.path.join(OUTPUT_FOLDER, unique_id + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{format_time(seg.start)} --> {format_time(seg.end)}\n{seg.text}\n\n")

    return send_file(srt_path, as_attachment=True)

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

if __name__ == "__main__":
    app.run()
