import pyaudio

# Initialize PyAudio
p = pyaudio.PyAudio()

# List available input devices
print("Input Devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"{i}: {device_info['name']}")

# List available output devices
print("\nOutput Devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"{i}: {device_info['name']}")

# Terminate PyAudio
p.terminate()
