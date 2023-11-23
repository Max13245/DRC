import pyroomacoustics as pra
from scipy.io import wavfile


class AcousticRoom:
    def __init__(self, room_dim, reverb, materials, audio) -> None:
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
        self.fs, self.audio = wavfile.read("assets/guiter_26k.wav")
        self.room = pra.ShoeBox(
            self.room_dim, materials=self.material, fs=self.fs, max_order=self.max_order
        )

    def add_speakers(self, speaker_positions):
        pass
