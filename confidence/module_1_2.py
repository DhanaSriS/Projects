import moviepy.editor as mp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()

# Function to perform speech-to-text conversion
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)  # Record the audio file
        text = recognizer.recognize_google(audio_data)  # Use Google Web Speech API to transcribe the audio
    return text

# Function to analyze text sentiment
def analyze_text_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    # Example confidence measurement logic
    confidence = (compound_score + 1) / 2  # Mapping [-1, 1] to [0, 1]

    # Scaling confidence to range between 0 and 100
    scaled_confidence = confidence * 100  # Mapping [0, 1] to [0, 100]
    return scaled_confidence

# Example usage
def main():
    video_path = "D:/C_files/Python/Confidence/video.mp4"
    audio_path = "D:/C_files/Python/Confidence/extracted_audio.wav"

    extract_audio(video_path, audio_path)
    transcribed_text = transcribe_audio(audio_path)
    confidence = analyze_text_sentiment(transcribed_text)

    print("Transcribed Text:", transcribed_text)
    print("Confidence Level:", confidence, "%")

if __name__ == "__main__":
    main()
