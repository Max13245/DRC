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
fs, audio_data = wavfile.read("../assets/KNOWER - Time Traveler.wav")

# Get the time values for each sample
time_values = np.arange(len(audio_data)) / fs

print(audio_data[int(fs * 10.4)])
print(len(audio_data))
print(fs)

print(audio_data)


# Plot the signal
plt.plot(time_values, audio_data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()


"""import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the audio signal
fs, audio_data = wavfile.read("../assets/guitar_16k.wav")

# Compute the FFT
fft_result = np.fft.fft(audio_data)
fft_freq = np.fft.fftfreq(len(fft_result), 1 / fs)

# Plot the frequency content
plt.plot(fft_freq, np.abs(fft_result))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Content of Audio Signal")
plt.show()"""

"""import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio signal
audio_file = "../assets/guitar_16k.wav"
y, sr = librosa.load(audio_file, sr=None)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert amplitude spectrogram to dB scale
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Display the spectrogram
librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Audio Signal")
plt.show()"""


"""import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio signal
audio_file = "../assets/guitar_16k.wav"
y, sr = librosa.load(audio_file, sr=None)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Display the spectrogram without converting to dB scale
librosa.display.specshow(np.abs(D), sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f")
plt.title("Magnitude Spectrogram of Audio Signal")
plt.show()"""


"""import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load the audio signal
audio_file = "../assets/guitar_16k.wav"
y, sr = librosa.load(audio_file, sr=None)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Extract the dominant frequency for each time frame
frequencies = librosa.amplitude_to_db(np.abs(D), ref=np.max).argmax(axis=0)
times = librosa.times_like(frequencies, sr=sr)

# Plot the dominant frequency over time
plt.plot(times, librosa.hz_to_midi(frequencies), label="Dominant Frequency")
plt.xlabel("Time (s)")
plt.ylabel("MIDI Note")
plt.title("Dominant Frequency Over Time")
plt.legend()
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Load the audio signal
audio_file = "../assets/guitar_16k.wav"
fs, audio_data = wavfile.read(audio_file)

# Compute the Short-Time Fourier Transform (STFT)
_, _, D = stft(audio_data, fs=fs, nperseg=1024)

# Extract the dominant frequency for each time frame
frequencies = np.argmax(np.abs(D), axis=0)
times = np.arange(len(frequencies)) * (len(audio_data) / fs)

# Plot the dominant frequency over time
plt.plot(times, frequencies, label="Dominant Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Frequency Bin Index")
plt.title("Dominant Frequency Over Time")
plt.legend()
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Load the audio signal
audio_file = "../assets/guitar_16k.wav"
fs, audio_data = wavfile.read(audio_file)

# Compute the Short-Time Fourier Transform (STFT)
_, _, D = stft(audio_data, fs=fs, nperseg=1024)

# Extract the dominant frequency for each time frame
frequencies = np.argmax(np.abs(D), axis=0)
times = np.arange(len(frequencies)) * (len(audio_data) / fs)

# Convert frequency bin indices to Hertz
freq_hz = frequencies * (fs / len(D))

# Plot the dominant frequency over time
plt.plot(times, freq_hz, label="Dominant Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Dominant Frequency Over Time")
plt.legend()
plt.show()"""
