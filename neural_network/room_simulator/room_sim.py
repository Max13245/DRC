import pyroomacoustics as pra
from scipy.io import wavfile
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


# Create named tuple
room_materials = namedtuple(
    "room_materials", ["ceiling", "floor", "north", "east", "south", "west"]
)

# Set seed so reproduceable
np.random.seed(100)


class AcousticRoom:
    def __init__(self, room_data) -> None:
        """Deleted reverb for now, might need later"""
        self.room_dim = eval(room_data[2])[1]
        self.speaker_positions = eval(room_data[2])[2]
        self.mic_position = eval(room_data[2])[0]
        self.directions = eval(room_data[2])[4]

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
        self.audio = self.remove_residue(self.audio)
        self.audio = self.get_random_sample(self.audio)

        # If dubble stream input get left of pair
        if type(self.audio[0]) != np.int16:
            self.audio = np.array([pair[0] for pair in self.audio])

        # Adjust master audio to master volume and remove last part (residue)
        self.master_volume = int(room_data[1])
        self.master_audio = self.adjust_to_master_volume()
        self.recorded_audio = None

        # Creating a room
        self.room = pra.ShoeBox(
            self.room_dim,
            materials=self.material,
            fs=self.fs,
            max_order=self.max_order,
            air_absorption=True,
        )

        # Use RT30 to get reflection levels
        self.init_RT30_sounds()
        self.reflection_levels = np.array([])
        self.RT30_reflection_test(eval(room_data[2])[2], eval(room_data[2])[0])

        # Threshold for significant peaks
        self.threshold = 30

    def add_speakers(self, streams) -> None:
        # One stream is the audio for that speaker
        for indx, position in enumerate(self.speaker_positions):
            # Create directivity for the speaker
            direction_object = pra.directivities.CardioidFamily(
                orientation=pra.directivities.DirectionVector(
                    azimuth=self.directions[indx], colatitude=15, degrees=True
                ),
                pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
            )

            self.room.add_source(
                position,
                signal=streams[indx],
                directivity=direction_object,
                delay=0,
            )

    def adjust_to_master_volume(self) -> np.array:
        # TODO: The amplitude seems to be unlineair so, account for that
        master_factor = self.master_volume / 100
        return (self.audio * master_factor).astype("int16")

    def get_normalized_fft(self, fft_sample: np.array) -> np.array:
        return fft_sample / len(fft_sample)

    def plot_fft_sample(
        self, frequencies: np.array, fft_amplitudes: np.array, normalized=True
    ) -> None:
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

    def plot_audio(self, audio: np.array) -> None:
        plt.plot(range(0, len(audio)), audio)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_audio_difference(self, audio1: np.array, audio2: np.array) -> None:
        plt.plot(range(0, len(audio1)), audio1)
        plt.plot(range(0, len(audio2)), audio2)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def get_fft_audio(self, audio: np.array) -> np.array:
        # For now devide by fs, but might be to large (sample by a whole num derived from fs)
        samples = np.array_split(audio, len(audio) / (self.fs / 100))

        fft_samples = []
        for sample in samples:
            # Compute the FFT
            fft_result = np.fft.fft(sample)

            # Each FFT element corresponds with all frequencies
            # So there are fft_result length * freqs length number of waves
            freqs = np.fft.fftfreq(len(sample), d=1 / self.fs)

            fft_samples.append((freqs, fft_result))

        return np.array(fft_samples)

    def remove_residue(self, audio: np.array):
        residue = len(audio) % self.fs
        audio = audio[:-residue]
        return audio

    def get_random_sample(self, audio: np.array):
        audio_duration = len(audio) / self.fs
        if audio_duration <= 10:
            return audio
        random_point = np.random.randint(0, int(audio_duration - 10)) * self.fs
        return audio[random_point : random_point + (10 * self.fs)]

    def get_ifft_audio(self, fft_amplitudes: list) -> np.array:
        # Compute the inverse of the (altered) fft for every sample
        # Use absolute to get only the amplitude
        reconstructed_wave = [np.fft.ifft(amplitude) for amplitude in fft_amplitudes]

        # Flatten the list from 2d to 1d
        concat_reconst_wave = np.concatenate(reconstructed_wave)
        return concat_reconst_wave

    def calculate_angle(self, y1, y2, y3):
        # Vectors representing the directions of the lines
        vec_AB = np.array([1, y2 - y1])
        vec_BC = np.array([1, y3 - y2])

        # Calculate the dot product
        dot_product = np.dot(vec_AB, vec_BC)

        # Calculate the magnitudes of the vectors
        magnitude_AB = np.linalg.norm(vec_AB)
        magnitude_BC = np.linalg.norm(vec_BC)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_AB * magnitude_BC)

        # Calculate the angle in radians
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        magnitude, angle_rad = np.abs(cos_theta), np.angle(cos_theta)
        # Use the angle in polar form
        theta_rad = angle_rad

        # Convert the angle to degrees
        theta_deg = np.degrees(theta_rad)

        return 180 - theta_deg

    def store_recorded_audio(self) -> bool:
        # Use angle between two lines
        self.recorded_audio = self.room.mic_array.signals[0]
        for indx in range(0, len(self.recorded_audio)):
            angle = self.calculate_angle(
                self.recorded_audio[indx],
                self.recorded_audio[indx + 1],
                self.recorded_audio[indx + 2],
            )
            if angle <= 90:
                self.recorded_audio = self.recorded_audio[
                    indx : len(self.master_audio) + indx
                ]
                break

            if indx > 2000:
                return False
        return True

    def get_significant_waves(self, amplitudes_sample):
        """Get the highest amplitude peaks with there frequency"""
        amplitudes_sample = np.delete(amplitudes_sample, 0)

        # Transform to absolute amplitudes
        amplitudes_sample = np.abs(amplitudes_sample)
        max_value = amplitudes_sample.max()
        division_factor = max_value / 100

        # In case division is 0
        if division_factor == 0:
            division_factor = 1

        # Add 1 to index, because 0 peak was deleted above
        indices = [
            indx + 1
            for indx, amplitude in enumerate(amplitudes_sample)
            if amplitude / division_factor >= self.threshold
        ]
        return indices

    def get_single_stream(self, dubble_stream):
        single_stream = [pair[0] for pair in dubble_stream]
        return single_stream

    def init_RT30_sounds(self) -> None:
        _, pink_noise_100 = wavfile.read("./neural_network/assets/Pink-noise-100Hz.wav")
        _, pink_noise_1000 = wavfile.read(
            "./neural_network/assets/Pink-noise-1000Hz.wav"
        )
        _, pink_noise_12500 = wavfile.read(
            "./neural_network/assets/Pink-noise-12500Hz.wav"
        )
        single_pink_noise_100 = self.get_single_stream(pink_noise_100)
        single_pink_noise_1000 = self.get_single_stream(pink_noise_1000)
        single_pink_noise_12500 = self.get_single_stream(pink_noise_12500)
        self.RT30_sounds = np.array(
            [single_pink_noise_100, single_pink_noise_1000, single_pink_noise_12500]
        )

    def RT30_reflection_test(self, speaker_positions, mic_position):
        # Test for every speaker and do 3 sweeptones (different sounds) of ... octave
        for position in speaker_positions:
            for sound in self.RT30_sounds:
                # Create a temporary sound source to get the reflection level for that speaker
                sound_source = pra.SoundSource(position, signal=sound, delay=0)
                self.room.add_source(sound_source)

                # Temporary mic
                self.room.add_microphone(mic_position, fs=self.fs)

                self.room.simulate(recompute_rir=True)

                # Get the rt30 and add to the reflection array
                RT30 = self.room.measure_rt60(decay_db=30)
                self.reflection_levels = np.append(self.reflection_levels, RT30[0])

                # Remove source and mic, because temporary
                self.room.sources.remove(sound_source)
                self.room.mic_array = None
