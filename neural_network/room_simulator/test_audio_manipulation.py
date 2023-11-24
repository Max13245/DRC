"""import soundfile as sf

# Load your original audio signal
original_signal, original_sr = sf.read("../assets/guitar_16k.wav")

# Define scaling factors
scale_factor_speaker1 = 0.5  # 50% of the original volume
scale_factor_speaker2 = 0.8  # 80% of the original volume

# Apply scaling to the audio signals
adjusted_signal_speaker1 = original_signal * scale_factor_speaker1
adjusted_signal_speaker2 = original_signal * scale_factor_speaker2

# Save the adjusted signals to new audio files
sf.write("mic_data/adjusted_audio_speaker1.wav", adjusted_signal_speaker1, original_sr)
sf.write("mic_data/adjusted_audio_speaker2.wav", adjusted_signal_speaker2, original_sr)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load an example audio file (replace 'your_audio_file.wav' with your file)
fs, audio_data = wavfile.read("../assets/guitar_16k.wav")

# Get the time values for each sample
time_values = np.arange(len(audio_data)) / fs

print(audio_data[int(fs * 10.4)])

# Plot the signal
plt.plot(time_values, audio_data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()
