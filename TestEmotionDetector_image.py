import cv2
import numpy as np
import os
from keras.models import model_from_json

# ================= EMOTION LABELS =================
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ================= LOAD MODEL =================
with open("emotion_model.json", "r") as json_file:
    model_json = json_file.read()

emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotion_model.h5")
print("✅ Emotion model loaded successfully")

# ================= IMAGE EMOTION FUNCTION =================
def File(input_path):

    # Output path (must be inside static/)
    output_dir = "static/test1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.jpg")

    # Read image
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Image not loaded")
        return input_path

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade CORRECTLY
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    # Detect faces (more sensitive settings)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("Faces detected:", len(faces))

    # If no face is detected
    if len(faces) == 0:
        cv2.putText(
            frame,
            "No Face Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = emotion_model.predict(roi_gray, verbose=0)
        emotion_label = emotion_dict[np.argmax(prediction)]

        # Draw emotion label
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Save output image
    cv2.imwrite(output_path, frame)

    # 🔥 MOST IMPORTANT LINE
    return output_path
