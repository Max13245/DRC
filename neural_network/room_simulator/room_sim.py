import pyroomacoustics as pra
from scipy.io import wavfile
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import time


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

    def get_normalized_fft(self, fft_sample):
        return fft_sample / len(fft_sample)

    def plot_fft_sample(self, frequencies, fft_amplitudes, normalized=True):
        if normalized:
            normalized_fft = self.get_normalized_fft(fft_amplitudes)
            plt.plot(
                frequencies[1 : len(frequencies)],
                np.abs(normalized_fft)[1 : len(normalized_fft)],
            )
        else:
            plt.plot(
                frequencies[1 : len(frequencies)],
                np.abs(fft_amplitudes)[1 : len(fft_amplitudes)],
            )

        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_audio(self, audio):
        plt.plot(range(0, len(audio)), audio)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def get_fft_audio(self):
        # For now devide by fs, but might be to large (sample by a whole num derived from fs)
        samples = np.array_split(
            self.master_audio, len(self.master_audio) / (self.fs / 100)
        )

        fft_samples = []
        for sample in samples:
            # Compute the FFT
            fft_result = np.fft.fft(sample)

            # Each FFT element corresponds with all frequencies
            # So there are fft_result length * freqs length number of waves
            freqs = np.fft.fftfreq(len(sample), d=1 / self.fs)

            fft_samples.append((freqs, fft_result))

        return fft_samples

    def get_ifft_audio(self, fft_amplitudes):
        # Compute the inverse of the (altered) fft for every sample
        # Use absolute to get only the amplitude
        reconstructed_wave = [np.fft.ifft(amplitude) for amplitude in fft_amplitudes]

        # Flatten the list from 2d to 1d
        concat_reconst_wave = np.concatenate(reconstructed_wave)
        return concat_reconst_wave
