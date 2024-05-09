#face and voice combined
import cv2
import moviepy.editor as mp
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load pre-trained models for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()

# Function to transcribe audio to text using Google's speech recognition service
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        # Use Google Web Speech API for transcription
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""  # If audio cannot be transcribed, return an empty string
    return text

# Function to measure confidence based on sentiment score
def measure_voice_confidence(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    # Example confidence measurement logic
    confidence = (compound_score + 1) / 2  # Mapping [-1, 1] to [0, 1]
    return confidence

# Function to measure confidence based on face analysis
def measure_face_confidence(video_path):
    cap = cv2.VideoCapture(video_path)
    total_confidence = 0
    num_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        num_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            num_eyes = len(eyes)
            confidence = min(num_eyes / 2, 1.0)
            total_confidence += confidence

    overall_confidence = total_confidence / num_frames if num_frames > 0 else 0
    return overall_confidence

# Ensemble method: Taking the average of confidence levels
def calculate_overall_confidence(face_confidence, voice_confidence):
    return (face_confidence + voice_confidence) / 2

# Example usage
def main():
    video_path = 'in_video.mp4'
    audio_path = 'extracted_audio.wav'

    extract_audio(video_path, audio_path)
    transcribed_text = transcribe_audio(audio_path)

    voice_confidence = measure_voice_confidence(transcribed_text)
    face_confidence = measure_face_confidence(video_path)

    overall_confidence = calculate_overall_confidence(face_confidence, voice_confidence)
    

    print("Overall Confidence:", round(overall_confidence * 100, 2), "%")

if __name__ == "__main__":
    main()
