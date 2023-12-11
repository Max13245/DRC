# create directivity object
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt

fs, audio = wavfile.read("./neural_network/assets/train_sounds/guitar_16k.wav")

room_dim = [8, 8, 3]
speaker_position = [2, 2, 1]
mic_position = [4, 4, 1.5]

room = pra.ShoeBox(
    room_dim,
    fs=fs,
    air_absorption=True,
)

dir_obj = CardioidFamily(
    orientation=DirectionVector(azimuth=45, colatitude=15, degrees=True),
    pattern_enum=DirectivityPattern.HYPERCARDIOID,
)

# place the source in the room
room.add_source(position=speaker_position, directivity=dir_obj, signal=audio)
room.add_microphone(mic_position)

room.simulate()

room.plot_rir()

plt.show()
