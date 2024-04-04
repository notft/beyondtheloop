import cv2
from deepface import DeepFace

desired_fps = 120


def detect_faces(video_path, eye_detected_count, eyes_not_focused_count):
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
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            if len(eyes) > 0:
                eye_detected_count += 1
            else:
                eyes_not_focused_count += 1

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

            emotion = analyze_emotion(frame)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            emotions_count[emotion] += 1

        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        frame_resized = cv2.resize(frame, (window_width, window_height))

        cv2.imshow('Emotion Detection', frame_resized)

        # Esc
        if cv2.waitKey(33) & 0xFF == 27:
            break

        total_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    emotions_percentage = {emotion: (count / total_frames) * 100 for emotion, count in emotions_count.items()}
    for emotion, percentage in emotions_percentage.items():
        print(f'{emotion}: {percentage:.2f}%')

    
    eyes_see = int(eye_detected_count)
    eyes_no_see =int(eyes_not_focused_count)
    total = eyes_see + eyes_no_see
    averageeyes = (eyes_see/total)*100
    print("AFP is: " + str(averageeyes) + "%")

def analyze_emotion(frame):
    res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = res[0]['dominant_emotion'] 
    return emotion

if __name__ == "__main__":
    window_width = 640
    window_height = 480
    video_path = r'C:\Users\aibel\Documents\New Harddrive\Personal\Hobbies\Code\Hackthon_ihalia\demo.mp4'
    eye_detected_count = 0
    eyes_not_focused_count = 0
    detect_faces(video_path, eye_detected_count, eyes_not_focused_count)
