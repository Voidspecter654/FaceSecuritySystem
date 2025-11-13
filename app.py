print("App starting...")

from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from model import db, User
import os
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face_security.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Create database and add hardcoded admin
with app.app_context():
    db.create_all()
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        hashed_pw = generate_password_hash('admin123', method='pbkdf2:sha256')
        admin = User(username='admin', password=hashed_pw, role='admin')
        db.session.add(admin)
        db.session.commit()


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for("admin"))
            else:
                return redirect(url_for('upload'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/admin')
@login_required
def admin():
    print("Admin route hit")  # keep for debugging
    if current_user.username != 'admin':
        return redirect(url_for('login'))
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)


@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')


@app.route('/test')
def test():
    return "<h1>Test route works!</h1>"


@app.route('/test_template')
def test_template():
    return render_template('admin_dashboard.html', users=[])



from flask import Response
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_MSMF)


import face_recognition
import pickle
import numpy as np
import time

# Load known encodings once at startup
print("[INFO] Loading known faces...")
with open("recognizer/encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Initialize camera
camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

import face_recognition
import numpy as np
import pickle
import time

# Load known encodings
print("[INFO] Loading known faces...")
with open("recognizer/encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Initialize video stream
camera = cv2.VideoCapture(1, cv2.CAP_MSMF)

def generate_frames():
    print("[INFO] Starting video stream with AI detection...")
    direction_text = ""

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            # Skip if frame not captured correctly
            continue

        # Ensure the frame is 8-bit and has 3 channels
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if len(frame.shape) == 2:
            # Grayscale to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Unexpected channel count
            continue

        # Resize frame for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

        # Detect faces
        try:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        except Exception as e:
            print(f"[WARNING] Face detection skipped: {e}")
            continue

        direction_text = ""  # Reset per frame
        frame_center = frame.shape[1] // 2  # Middle X of frame

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Scale back up face locations
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            color = (0, 0, 255) if name != "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name == "Unknown":
                face_center_x = (left + right) // 2
                if face_center_x < frame_center - 80:
                    direction_text = "MOVE DRONE LEFT ←"
                elif face_center_x > frame_center + 80:
                    direction_text = "MOVE DRONE RIGHT →"
                else:
                    direction_text = "MOVE FORWARD ↑"

        if direction_text:
            cv2.putText(frame, direction_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
