import pyroomacoustics as pra
from scipy.io import wavfile
from collections import namedtuple
import numpy as np


# Create named tuple
room_materials = namedtuple(
    "room_materials", ["ceiling", "floor", "north", "east", "south", "west"]
)


class AcousticRoom:
    def __init__(self, room_data) -> None:
        """Deleted reverb for now, might need later"""
        self.room_dim = eval(room_data[2])[1]
        self.speaker_positions = room_data[2][2]
        self.mic_position = room_data[2][0]

        # Create a namedtuple for materials TODO: Put dict as csv structure
        matrls = [material for material in eval(room_data[2])[3]]
        materials = room_materials(
            matrls[0], matrls[1], matrls[2], matrls[3], matrls[4], matrls[5]
        )
        self.material = pra.make_materials(
            ceiling=materials.ceiling,
            floor=materials.floor,
            north=materials.north,
            east=materials.east,
            south=materials.south,
            west=materials.west,
        )

        self.max_order = 1  # TODO: Is default, can be calculated with sabine formula?
        self.fs, self.audio = wavfile.read(room_data[0])
        self.master_audio = np.array(
            self.adjust_to_master_volume(int(room_data[1])), dtype="int16"
        )

        # Creating a room
        self.room = pra.ShoeBox(
            self.room_dim,
            materials=self.material,
            fs=self.fs,
            max_order=self.max_order,
            air_absorption=True,
        )

    def add_speakers(self, speaker_props) -> None:
        for speaker in speaker_props:
            # speaker[0] is the location of the speaker and speaker[1] is the audio for that speaker
            self.room.add_source(speaker[0], signal=speaker[1], delay=0)

    def add_mic(self, position: tuple) -> None:
        self.room.add_microphone(position)

    def add_mics(self, positions: list) -> None:
        # Positions must be following size: (dim, n_mics)
        self.room.add_microphone_array(positions)

    def adjust_to_master_volume(self, master_percentage):
        # TODO: The amplitude seems to be unlineair so, account for that
        master_factor = master_percentage / 100
        return (self.audio * master_factor).astype("int16")

    def get_fft_audio(self):
        # For now devide by fs, but might be to large (sample by a whole num derived from fs)
        # TODO: Deside if sampling is better for quality
        # samples = np.array_split(self.master_audio, len(self.master_audio) / self.fs)

        # Compute the FFT
        fft_result = np.fft.fft(self.master_audio)

        # Each FFT element corresponds with all frequencies
        # So there are fft_result length * freqs length number of waves
        freqs = np.fft.fftfreq(len(self.master_audio), d=1 / self.fs)
        return (fft_result, freqs)

    def get_desampled_audio(self):
        pass
