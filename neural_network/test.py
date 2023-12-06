import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
duration = 1.0  # Duration in seconds
sample_rate = 44100  # Sample rate in Hertz

# Time values
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sinusoidal waves
freq1 = 1000
freq2 = 1100
wave1 = np.sin(2 * np.pi * freq1 * t)
wave2 = np.sin(2 * np.pi * freq2 * t)

# Combine the waves
combined_wave = wave1 + wave2

# Compute the FFT
fft_result = np.fft.fft(combined_wave)

# Get the corresponding frequencies
freqs = np.fft.fftfreq(len(combined_wave), d=1 / sample_rate)

counter = 0
for result in np.abs(fft_result):
    if result < 1e-8:
        counter += 1
    else:
        print(result)

print(counter)
print("Difference: " + str(sample_rate - counter))

# Plot the combined wave
plt.subplot(4, 1, 1)
plt.plot(t, combined_wave)
plt.title("Combined Soundwave (1000 Hz + 1100 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot the individual waves
plt.subplot(4, 1, 2)
plt.plot(t, wave1)
plt.title("Individual Wave (1000 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(t, wave2)
plt.title("Individual Wave (1100 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot the FFT result
plt.subplot(4, 1, 4)
plt.plot(freqs, np.abs(fft_result))
plt.title("FFT Result (Positive Frequencies)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)  # Set the x-axis limit to focus on the positive frequencies

plt.tight_layout()
plt.show()
