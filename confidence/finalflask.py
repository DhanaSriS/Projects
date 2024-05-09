from flask import Flask, render_template, request, jsonify
import os
import cv2
import moviepy.editor as mp
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Extract audio
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_audio.wav')
    extract_audio(video_path, audio_path)

    # Transcribe audio
    transcribed_text = transcribe_audio(audio_path)

    # Measure voice confidence
    voice_confidence = measure_voice_confidence(transcribed_text)

    # Measure face confidence
    face_confidence = measure_face_confidence(video_path)

    # Calculate overall confidence
    overall_confidence = calculate_overall_confidence(face_confidence, voice_confidence)

    return jsonify({'confidence': overall_confidence})

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
