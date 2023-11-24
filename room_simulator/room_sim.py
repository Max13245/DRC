import pyroomacoustics as pra
from scipy.io import wavfile


class AcousticRoom:
    def __init__(self, room_dim, reverb, materials, audi_path) -> None:
        self.room_dim = room_dim
        self.reverbaration_time = reverb
        self.material = pra.make_materials(
            ceiling=materials.ceiling,
            floor=materials.floor,
            east=materials.east,
            west=materials.west,
            north=materials.north,
            south=materials.south,
        )
        self.max_order = 1  # TODO: Is default
        self.fs, self.audio = wavfile.read(audi_path)

        # Creating a room
        self.room = pra.ShoeBox(
            self.room_dim,
            materials=self.material,
            fs=self.fs,
            max_order=self.max_order,
            air_absorption=True,
        )

    def add_speakers(self, speaker_positions) -> None:
        for speaker in speaker_positions:
            self.room.add_source(speaker, signal=self.audio, delay=0)

    def add_mic(self, position) -> None:
        self.room.add_microphone(position)

    def add_mics(self, positions) -> None:
        # Positions must be following size: (dim, n_mics)
        self.room.add_microphone_array(positions)
