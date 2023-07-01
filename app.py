import os
from flask import Flask, render_template, send_from_directory, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from src.parking_space_classifier import ParkClassifier
from src.car_park_coordinate_generator import CoordinateDenoter

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "mp4"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

classifier = ParkClassifier("C:/Users/yawir/Downloads/car-parking-finder-main/data/source/CarParkPos")
coordinate_generator = CoordinateDenoter()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            if filename.lower().endswith((".mp4")):
                result_path = process_video(file_path)
                return redirect(url_for("static", filename=result_path))

            result_path = process_image(file_path)
            return redirect(url_for("static", filename=result_path))
    
    return render_template("index.html")

@app.route("/result/<path:filename>")
def get_image(filename):
    return send_from_directory('images', filename)
# def result(filename):
#     result_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     return render_template("result.html", result_filename=result_path)

def process_image(file_path):
    image = cv2.imread(file_path)
    processed_image = classifier.implement_process(image)
    result_image = classifier.classify(image, processed_image)
    result_filename = f"result_{os.path.basename(file_path)}"
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
    static_url = url_for('static', filename=os.path.normpath(result_path))
    cv2.imwrite(result_path, result_image)
    return result_filename

def process_video(file_path):
    video = cv2.VideoCapture(file_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    result_filename = f"result_{os.path.basename(file_path)}"
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        processed_frame = classifier.implement_process(frame)
        result_frame = classifier.classify(frame, processed_frame)
        out.write(result_frame)

    video.release()
    out.release()
    return result_filename

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    coordinate_generator.read_positions()
    # coordinate_generator.demonstration()

    app.run(debug=True)
