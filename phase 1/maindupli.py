import cv2
from cv_operations import detect_faces
from emotion_detection import analyze_emotion


class AttendanceApp:
    def __init__(self, video_path):
        self.video_path = video_path

    def start_detection(self):
        for frame, faces in detect_faces(self.video_path):
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    emotion = analyze_emotion(frame)

                    cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(33) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r'C:\Users\aibel\Documents\New Harddrive\Personal\Hobbies\Code\Hackthon_ihalia\demo.mp4'
    app = AttendanceApp(video_path)
    app.start_detection()
