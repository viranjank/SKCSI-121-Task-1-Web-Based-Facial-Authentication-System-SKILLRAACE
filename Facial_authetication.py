from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import threading

app = Flask(__name__)

KNOWN_FACES_DIR = 'known_faces'
known_faces = {}

# Global variables for timer
timer_seconds = 10
timer_active = False


def load_known_faces():
    # Load known faces from a directory
    for filename in os.listdir(KNOWN_FACES_DIR):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        known_faces[name] = cv2.imread(image_path)


def save_image(image, filename):
    image_path = os.path.join(KNOWN_FACES_DIR, f"{filename}.jpg")
    cv2.imwrite(image_path, image)


def recognize_face(frame):
    # Compare the unknown face with known faces
    for name, known_face in known_faces.items():
        if is_match(frame, known_face):
            return name
    return "Unknown"


def start_timer(frame):
    global timer_seconds
    global timer_active
    timer_active = True
    while timer_seconds > 0:
        timer_seconds -= 1
        cv2.putText(frame, f'Time Left: {timer_seconds} seconds', (
            20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Capturing...', frame)
        cv2.waitKey(1000)  # Wait for 1 second
    timer_active = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global timer_seconds
        global timer_active
        # Start timer thread
        timer_seconds = 10
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        timer_thread = threading.Thread(target=start_timer, args=(frame,))
        timer_thread.start()

        # Capture user's face image from camera
        while timer_active and ret:
            ret, frame = camera.read()
            cv2.putText(frame, f'Time Left: {timer_seconds} seconds', (
                20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Capturing...', frame)
            # Wait for a very short time to allow the timer thread to continue
            cv2.waitKey(1)
        camera.release()
        cv2.destroyAllWindows()

        if not ret:
            return "Failed to capture image from camera!"

        name = request.form['name']
        save_image(frame, name)
        known_faces[name] = frame
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        global timer_seconds
        global timer_active
        # Start timer thread
        timer_seconds = 10
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        timer_thread = threading.Thread(target=start_timer, args=(frame,))
        timer_thread.start()

        # Capture user's face image from camera
        while timer_active and ret:
            ret, frame = camera.read()
            cv2.putText(frame, f'Time Left: {timer_seconds} seconds', (
                20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Capturing...', frame)
            # Wait for a very short time to allow the timer thread to continue
            cv2.waitKey(1)
        camera.release()
        cv2.destroyAllWindows()

        if not ret:
            return "Failed to capture image from camera!"

        name = recognize_face(frame)
        if name != "Unknown":
            return "Login successful"
        else:
            return "Login failed"

    return render_template('login.html')


def is_match(face_image, known_face):
    # Perform face detection and feature extraction using OpenCV
    # Compare the features to determine if it's a match
    # Here, you'll need to implement face detection and feature extraction using OpenCV
    # You can use techniques like Haar cascades for face detection
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 1:  # Assuming only one face is detected
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        # Perform feature extraction and comparison
        # Return True if it's a match, False otherwise
        # Here, you would typically use more sophisticated feature extraction and comparison techniques
        return True
    return False


if __name__ == '__main__':
    load_known_faces()
    app.run(debug=True)
