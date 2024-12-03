import librosa
import os
import matplotlib.pyplot as plt

# Directory containing sound files
sound_dir = "Sounds"

# List all files in the directory
sound_files = [os.path.join(sound_dir, file) for file in os.listdir(sound_dir) if file.endswith(".wav")]

# Load each sound file
audio_data = {}
for file in sound_files:
    data, sr = librosa.load(file, sr=None)
    audio_data[file] = (data, sr)
    print(f"Loaded {file}: {len(data)} samples at {sr} Hz")

# Visualize the waveforms
for file, (data, sr) in audio_data.items():
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(f"Waveform of {os.path.basename(file)}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
 

