import cv2
from cv_operations import detect_faces
from emotion_detection import analyze_emotion

window_width = 640
window_height = 480

cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Emotion Detection', window_width, window_height)

video_path = r'C:\Users\aibel\Documents\New Harddrive\Personal\Hobbies\Code\Ihalia_hackathon\phase 1\demo.mp4'

emotions_count = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0
}

eye_detected_count = 0
eyes_not_focused_count = 0
total_frames = 0

for frame, faces, eye_detected_count, eyes_not_focused_count in detect_faces(video_path, eye_detected_count, eyes_not_focused_count):
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

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

cv2.destroyAllWindows()

emotions_percentage = {emotion: (count / total_frames) * 100 for emotion, count in emotions_count.items()}

for emotion, percentage in emotions_percentage.items():
    print(f'{emotion}: {percentage:.2f}%')

#print(f'eyes see times: {eye_detected_count}')
#print(f'eye not see times : {eyes_not_focused_count}')

#avg eyes
eyes_see = int(eye_detected_count)
eyes_no_see =int(eyes_not_focused_count)
total = eyes_see + eyes_no_see
averageeyes = (eyes_see/total)*100
print("AFP is: " + str(averageeyes) + "%")
if averageeyes > 45:
    print("Student was attendive in class")
else:
    print("Student wasnt attendive in class")



