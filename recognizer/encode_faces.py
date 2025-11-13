import os
import face_recognition
import pickle
import cv2

# Path to the dataset
dataset_path = "static/uploads"
encodings_file = "recognizer/encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

for user_folder in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user_folder)
    if not os.path.isdir(user_path):
        continue

    for image_name in os.listdir(user_path):
        image_path = os.path.join(user_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(user_folder)

print(f"[INFO] Encoded {len(known_encodings)} faces.")

# Save the encodings
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encodings saved to {encodings_file}")
