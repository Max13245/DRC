class SoundControl:
    def __init__(self) -> None:
        pass


"""import pyaudio
import wave

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get("deviceCount")
device_names = info.get("deviceName")

print(info)
print("\n")
print(numdevices)
print("\n")
print(device_names)

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get("maxOutputChannels")) > 0:
        print(
            "Output Device id ",
            i,
            " - ",
            p.get_device_info_by_host_api_device_index(0, i).get("name"),
        )

filename = "KNOWER - Time Traveler.wav"

# Set chunk size of 1024 samples per data frame
chunk = 1024

# Open the sound file
wf = wave.open(filename, "rb")

# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True,
    output_device_index=0,
)

# Read data in chunks
data = wf.readframes(chunk)

# Play the sound by writing the audio data to the stream
while data != "":
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()"""
