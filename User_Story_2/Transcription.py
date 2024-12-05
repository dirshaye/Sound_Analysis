from google.cloud import speech
import os
from pydub import AudioSegment
from pydub.playback import play

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-key.json"

def preprocess_audio(input_path, output_path, target_sample_rate=16000):
    """
    Preprocesses an audio file by converting it to mono and resampling to the target sample rate.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_path) 

    # Convert to mono
    audio = audio.set_channels(1)

    # Resample to the target sample rate
    audio = audio.set_frame_rate(target_sample_rate)

    # Export the preprocessed audio
    audio.export(output_path, format="wav")
    return output_path

def transcribe_audio(file_path, output_file="Transcriptions/transcription.txt"):
    """
    Transcribes an audio file, combines all results, counts the total words,
    and saves the transcription to a text file.
    """
    client = speech.SpeechClient()

    # Load audio file
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    # Configure the request
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    # Perform transcription
    response = client.recognize(config=config, audio=audio)

    # Combine and process results
    full_transcript = " ".join(result.alternatives[0].transcript for result in response.results)
    print("Full Transcript:", full_transcript)

    # Count total words
    word_count = len(full_transcript.split())
    print("Total Word Count:", word_count)

    # Save transcription to a file
    with open(output_file, "w") as file:
        file.write(full_transcript)
    print(f"Transcription saved to {output_file}")

# Example usage
raw_audio_path = "Sounds/english_1.wav"
processed_audio_path = "Sounds/english_1_processed.wav"

# Preprocess the audio
preprocessed_file = preprocess_audio(raw_audio_path, processed_audio_path)

# Transcribe the preprocessed audio
transcribe_audio(preprocessed_file)
