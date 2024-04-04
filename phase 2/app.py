from flask import Flask, render_template, request
import cv2
from deepface import DeepFace

app = Flask(__name__)

def detect_faces(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(video_path)

    emotions_count = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0
    }

    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Emotion analysis
            res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = res[0]['dominant_emotion']
            emotions_count[emotion.lower()] += 1

        total_frames += 1

    cap.release()
    
    emotions_percentage = {emotion: (count / total_frames) * 100 for emotion, count in emotions_count.items()}
    return emotions_percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        video_file = request.files['video']
        video_path = 'temp_video.mp4'
        video_file.save(video_path)
        emotions_info = detect_faces(video_path)
        return render_template('result.html', emotions_info=emotions_info)

if __name__ == '__main__':
    app.run(debug=True)
