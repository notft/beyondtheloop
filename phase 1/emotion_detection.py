from deepface import DeepFace

def analyze_emotion(frame):
    
    res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    
    emotion = res[0]['dominant_emotion'] 
    
    return emotion
