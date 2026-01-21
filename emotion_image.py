import cv2
from deepface import DeepFace
import os

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def File(input_path):
    # Create output path in the same folder as input
    folder = os.path.dirname(input_path)
    output_path = os.path.join(folder, "out.jpg")

    # Load the image
    frame = cv2.imread(input_path)
    if frame is None:
        print("⚠️ Error: Cannot load image from", input_path)
        return None

    # Resize image for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and RGB
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("⚠️ No face detected in image.")
        cv2.imwrite(output_path, frame)
        return output_path

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        # ⚡ Use faster detector backend
        result = DeepFace.analyze(
            face_roi,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'  # Faster backend
        )

        # Extract dominant emotion
        emotion = result[0]['dominant_emotion']
        print("Detected emotion:", emotion)

        # Draw bounding box + emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save processed image
    cv2.imwrite(output_path, frame)
    print("✅ Output saved at:", output_path)
    return output_path


