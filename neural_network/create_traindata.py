from csv import writer
from random import randint, uniform, choice, sample
from os import listdir

# Everything in meters
MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH = 3, 10
BLOCKS = 3
MAX_SPEAKER_HEIGHT = 2
MAX_PERTURBATION = 0.5
TRAIN_SOUNDS_DIR = "./neural_network/assets/train_sounds"
MATERIALS = [
    "hard_surface",
    "brickwork",
    "rough_concrete",
    "unpainted_concrete",
    "rough_lime_wash",
    "smooth_brickwork_flush_pointing",
    "smooth_brickwork_10mm_pointing",
    "brick_wall_rough",
    "ceramic_tiles",
    "limestone_wall",
    "reverb_chamber",
    "concrete_floor",
    "marble_floor",
    "plasterboard",
    "wooden_lining",
    "wood_1.6cm",
    "plywood_thin",
    "wood_16mm",
    "audience_floor",
    "stage_floor",
    "wooden_door",
]

train_data = []
for filename in listdir(TRAIN_SOUNDS_DIR):
    train_data_single_sound = []
    for x in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
        for y in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
            for z in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
                for x_block in range(BLOCKS):
                    x_block_length = x / BLOCKS
                    y_block_length = y / BLOCKS
                    z_block_length = z / BLOCKS
                    for y_block in range(BLOCKS):
                        for z_block in range(BLOCKS):
                            maximum_reach_x = x_block_length * (x_block + 1)
                            minimum_reach_x = maximum_reach_x - x_block_length
                            mic_random_x_position = round(
                                uniform(minimum_reach_x, maximum_reach_x), 2
                            )
                            maximum_reach_y = y_block_length * (y_block + 1)
                            minimum_reach_y = maximum_reach_y - y_block_length
                            mic_random_y_position = round(
                                uniform(minimum_reach_y, maximum_reach_y), 2
                            )
                            maximum_reach_z = z_block_length * (z_block + 1)
                            minimum_reach_z = maximum_reach_z - z_block_length
                            mic_random_z_position = round(
                                uniform(minimum_reach_z, maximum_reach_z), 2
                            )
                            speaker_height = round(uniform(0, MAX_SPEAKER_HEIGHT), 2)
                            speaker_height_1 = speaker_height + round(
                                uniform(-MAX_PERTURBATION, MAX_PERTURBATION),
                                2,
                            )
                            if speaker_height_1 < 0:
                                speaker_height_1 = 0
                            speaker_height_2 = speaker_height + round(
                                uniform(-MAX_PERTURBATION, MAX_PERTURBATION),
                                2,
                            )
                            if speaker_height_2 < 0:
                                speaker_height_2 = 0
                            speaker_height_3 = speaker_height + round(
                                uniform(-MAX_PERTURBATION, MAX_PERTURBATION),
                                2,
                            )
                            if speaker_height_3 < 0:
                                speaker_height_3 = 0
                            speaker_height_4 = speaker_height + round(
                                uniform(-MAX_PERTURBATION, MAX_PERTURBATION),
                                2,
                            )
                            if speaker_height_4 < 0:
                                speaker_height_4 = 0
                            speaker_height_5 = speaker_height + round(
                                uniform(-MAX_PERTURBATION, MAX_PERTURBATION),
                                2,
                            )
                            if speaker_height_5 < 0:
                                speaker_height_5 = 0

                            train_sample = [
                                f"./neural_network/assets/train_sounds/{filename}",
                                randint(1, 100),
                                [
                                    (
                                        mic_random_x_position,
                                        mic_random_y_position,
                                        mic_random_z_position,
                                    ),  # Mic position
                                    (x, y, z),  # Room measurements
                                    [
                                        (
                                            round(uniform(0, x / 2), 2),
                                            round(uniform(0, y / 2), 2),
                                            speaker_height_1,
                                        ),  # Speaker 1
                                        (
                                            round(uniform(0, x / 2), 2),
                                            round(uniform(y / 2, y), 2),
                                            speaker_height_2,
                                        ),  # Speaker 2
                                        (
                                            round(uniform(x / 2, x), 2),
                                            round(uniform(y / 2, y), 2),
                                            speaker_height_3,
                                        ),  # Speaker 3
                                        (
                                            round(uniform(x / 2, x), 2),
                                            round(uniform(0, y / 2), 2),
                                            speaker_height_4,
                                        ),  # Speaker 4
                                        (
                                            randint(0, x),
                                            randint(0, y),
                                            speaker_height_5,
                                        ),  # Subwoofer
                                    ],  # Speaker positions
                                    [
                                        choice(MATERIALS) for _ in range(6)
                                    ],  # 6 Materials
                                    [
                                        round(uniform(0, 90), 2),
                                        round(uniform(90, 180), 2),
                                        round(uniform(180, 270), 2),
                                        round(uniform(270, 360), 2),
                                        round(uniform(0, 360), 2),
                                    ],
                                ],
                            ]
                            train_data_single_sound.append(train_sample)
    sampled_train_data = sample(train_data_single_sound, 200)
    train_data += sampled_train_data

with open("./neural_network/train_data_input.csv", "w") as csvfile:
    # Create a writer object to write to the file
    csvwriter = writer(csvfile)

    # writing the data rows
    csvwriter.writerows(train_data)
