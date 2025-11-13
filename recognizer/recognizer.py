import cv2
import face_recognition
import numpy as np
import os
from flask_socketio import SocketIO

socketio = SocketIO()

# Load known encodings
def load_known_faces(encoding_path="recognizer/encodings.npy", names_path="recognizer/names.npy"):
    if os.path.exists(encoding_path) and os.path.exists(names_path):
        known_encodings = np.load(encoding_path, allow_pickle=True)
        known_names = np.load(names_path, allow_pickle=True)
        return known_encodings, known_names
    return [], []

known_encodings, known_names = load_known_faces()

def generate_frames():
    camera = cv2.VideoCapture(1)
    sample_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        possible_path = f"static/uploads/sample{ext}"
        if os.path.exists(possible_path):
            sample_path = possible_path
            break

    if not sample_path:
        print("[ERROR] No sample image found in static/uploads/")
        return

    frame = cv2.imread(sample_path)
    if frame is None:
        print("[ERROR] Failed to load sample image.")
        return

    print("[INFO] Running face recognition on sample image...")

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    command = "Hover"

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        face_names.append(name)

        # Scale coordinates back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Determine movement direction based on face position
        face_center_x = (left + right) // 2
        frame_center_x = frame.shape[1] // 2

        if name != "Unknown":
            if face_center_x < frame_center_x - 50:
                command = "Move Left"
            elif face_center_x > frame_center_x + 50:
                command = "Move Right"
            else:
                command = "Follow"

    # Emit command (for simulated drone)
    socketio.emit("drone_command", {"command": command})

    print(f"[RESULT] Face(s): {face_names}, Command: {command}")

    # Save result frame for verification
    cv2.imwrite("static/alerts/result_preview.jpg", frame)
    print("[INFO] Result saved to static/alerts/result_preview.jpg")

    yield b""
