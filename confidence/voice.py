import librosa
from google.cloud import speech_v1p1beta1
from collections import defaultdict

# Load audio file
audio_file = 'audio.wav'
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Transcribe speech
client = speech_v1p1beta1.SpeechClient()
response = client.recognize(config={"encoding": "LINEAR16", "sample_rate_hertz": sample_rate}, audio={"content": audio_data})
transcript = response.results[0].alternatives[0].transcript

# Define fillers
fillers = ["um", "uh", "like"]  # Add more fillers as needed

# Detect pauses and fillers
pauses = []
filler_counts = defaultdict(int)
for word_info in response.results[0].alternatives[0].words:
    word = word_info.word.lower()
    if word in fillers:
        filler_counts[word] += 1
    elif word == '':
        pauses.append((word_info.start_time.seconds + word_info.start_time.nanos * 1e-9,
                       word_info.end_time.seconds + word_info.end_time.nanos * 1e-9))

# Calculate total duration of pauses
total_pause_duration = sum(end - start for start, end in pauses)

# Calculate total filler counts
total_filler_count = sum(filler_counts.values())

# Calculate total speech duration
total_speech_duration = audio_data.shape[0] / sample_rate

# Calculate pause rate (pause duration / total speech duration)
pause_rate = total_pause_duration / total_speech_duration

# Calculate filler rate (filler count / total speech duration)
filler_rate = total_filler_count / total_speech_duration

# Print results
print("Total Pause Duration:", total_pause_duration, "seconds")
print("Total Filler Count:", total_filler_count)
print("Total Speech Duration:", total_speech_duration, "seconds")
print("Pause Rate:", pause_rate)
print("Filler Rate:", filler_rate)
