import cv2 
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
desired_fps = 120
cap = cv2.VideoCapture(0) 

emotions_count = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0}

eye_detected_count = 0
eyes_not_focused_count = 0

while True:
    _, frame = cap.read()
    
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
       
        res = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
        
        if res:
            emotion = res[0]['dominant_emotion']
            emotions_count[emotion] += 1
            
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

total_faces = sum(emotions_count.values())
emotions_percentage = {emotion: (count / total_faces) * 100 for emotion, count in emotions_count.items()}

for emotion, percentage in emotions_percentage.items():
    print(f'{emotion}: {percentage:.2f}%')

print(f'Total number of times eyes were detected: {eye_detected_count}')
print(f'Total number of times eyes were not focused on the screen: {eyes_not_focused_count}')