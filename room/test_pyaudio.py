import time
import numpy as np
import pyaudio

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define parameters
sample_rate = 44100
volume = 0.5  # Adjust as needed

outputs = [1, 2]

# Define the number of channels
num_channels = 2

# Open individual streams for each DAC
streams = [
    p.open(
        format=pyaudio.paFloat32,
        channels=2,  # Each stream handles a single channel
        rate=sample_rate,
        output=True,
        output_device_index=outputs[i],
    )
    for i in range(num_channels)
]


def calculate_loudness_percentage():
    # Your loudness calculation logic here
    # This function should return a value between 0 and 100 for each channel
    # Example: return left_percentage, right_percentage, subwoofer_percentage
    return (20, 40, 60, 20, 60, 40)


def percentage_to_amplitude(percentage):
    # Convert percentage to amplitude
    return percentage / 100.0 * volume


n = 0
while True:
    n += 1
    (
        dac1_left_percentage,
        dac1_right_percentage,
        dac2_left_percentage,
        dac2_right_percentage,
        dac3_left_percentage,
        dac3_right_percentage,
    ) = calculate_loudness_percentage()

    # Convert percentage to amplitude
    dac1_left_amplitude = percentage_to_amplitude(dac1_left_percentage)
    dac1_right_amplitude = percentage_to_amplitude(dac1_right_percentage)
    dac2_left_amplitude = percentage_to_amplitude(dac2_left_percentage)
    dac2_right_amplitude = percentage_to_amplitude(dac2_right_percentage)
    dac3_left_amplitude = percentage_to_amplitude(dac3_left_percentage)
    dac3_right_amplitude = percentage_to_amplitude(dac3_right_percentage)

    # Generate a simple sine wave as an example for each channel
    t = np.arange(0, 1, 1.0 / sample_rate)

    dac1_left_wave = dac1_left_amplitude * np.sin(2 * np.pi * 440 * t)
    dac1_right_wave = dac1_right_amplitude * np.sin(2 * np.pi * 440 * t)
    dac2_left_wave = dac2_left_amplitude * np.sin(2 * np.pi * 440 * t)
    dac2_right_wave = dac2_right_amplitude * np.sin(2 * np.pi * 440 * t)
    dac3_left_wave = dac3_left_amplitude * np.sin(2 * np.pi * 440 * t)
    dac3_right_wave = dac3_right_amplitude * np.sin(2 * np.pi * 440 * t)

    # Combine the waves for each DAC
    combined_wave1 = np.column_stack((dac1_left_wave, dac1_right_wave))
    combined_wave2 = np.column_stack((dac2_left_wave, dac2_right_wave))
    combined_wave3 = np.column_stack((dac3_left_wave, dac3_right_wave))

    # Convert to bytes
    audio_data1 = (combined_wave1 * 32767).astype(np.int16).tobytes()
    audio_data2 = (combined_wave2 * 32767).astype(np.int16).tobytes()
    audio_data3 = (combined_wave3 * 32767).astype(np.int16).tobytes()

    list_audio_data = [audio_data1, audio_data2, audio_data3]

    # Output to the DACs
    for i in range(num_channels):
        streams[i].write(list_audio_data[i])

    time.sleep(0.1)  # Adjust sleep duration as needed

    if n > 1000:
        break

# Close streams and terminate PyAudio
for stream in streams:
    stream.stop_stream()
    stream.close()

p.terminate()
