from flask import Flask, render_template, request, send_file
from faster_whisper import WhisperModel
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Optimized Whisper model for FREE hosting (important)
model = WhisperModel(
    "small",
    compute_type="int8",
    cpu_threads=2
)

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:06.3f}".replace('.', ',')

def generate_srt(segments, filename):
    srt_path = f"{filename}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f.write(seg.text + "\n\n")
    return srt_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    unique_name = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, unique_name + "_" + file.filename)
    file.save(filepath)

    segments, info = model.transcribe(filepath)
    srt_file = generate_srt(segments, filepath)

    return send_file(srt_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
