from flask import Flask, request, render_template
import os

from emotion_image import File as DeepFaceFile   # DeepFace image emotion
from TestEmotionDetector_image import File as CNNFile  # CNN image emotion

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------- DEEPFACE IMAGE ----------------
@app.route('/analyse', methods=['POST'])
def analyse():

    file = request.files.get('file')

    if not file or file.filename == "":
        return render_template('index.html', error="No file selected")

    upload_folder = "static/test"
    os.makedirs(upload_folder, exist_ok=True)

    input_path = os.path.join(upload_folder, file.filename)
    file.save(input_path)

    # DeepFace emotion detection
    output_path = DeepFaceFile(input_path)

    return render_template(
        'index.html',
        image1=input_path,
        image2=output_path
    )


# ---------------- DEEPFACE LIVE ----------------
@app.route('/Live')
def Live():
    from emotion import Start
    Start(0)
    return render_template('index.html')


# ---------------- CNN IMAGE ----------------
@app.route('/analyse1', methods=['POST'])
def analyse1():

    file = request.files.get('file')

    if not file or file.filename == "":
        return render_template('index.html', error="No file selected")

    upload_folder = "static/test1"
    os.makedirs(upload_folder, exist_ok=True)

    input_path = os.path.join(upload_folder, file.filename)
    file.save(input_path)

    # CNN emotion detection
    output_path = CNNFile(input_path)

    return render_template(
        'index.html',
        image3=input_path,
        image4=output_path
    )


# ---------------- CNN LIVE ----------------
@app.route('/Live11')
def Live11():
    from TestEmotionDetector import Start
    Start(0)
    return render_template('index.html')


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)



