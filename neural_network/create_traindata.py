from csv import writer
from random import randint, uniform, choice
from os import listdir

# Everything in meters
MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH = 3, 20
X_BLOCKS, Y_BLOCKS, Z_BLOCKS = 3, 3, 3
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
    for x in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
        for y in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
            for z in range(MINIMUM_ROOM_LENGTH, MAXIMUM_ROOM_LENGTH):
                for x_block in range(X_BLOCKS):
                    for y_block in range(Y_BLOCKS):
                        for z_block in range(Z_BLOCKS):
                            x_block_length = x / X_BLOCKS
                            maximum_reach_x = x_block_length * x_block
                            minimum_reach_x = maximum_reach_x - x_block_length
                            mic_random_x_position = round(
                                uniform(minimum_reach_x, maximum_reach_x), 2
                            )
                            y_block_length = y / Y_BLOCKS
                            maximum_reach_y = y_block_length * y_block
                            minimum_reach_y = maximum_reach_y - y_block_length
                            mic_random_y_position = round(
                                uniform(minimum_reach_y, maximum_reach_y), 2
                            )
                            z_block_length = z / Z_BLOCKS
                            maximum_reach_z = z_block_length * z_block
                            minimum_reach_z = maximum_reach_z - z_block_length
                            mic_random_z_position = round(
                                uniform(minimum_reach_z, maximum_reach_z), 2
                            )
                            speaker_height = round(uniform(0, MAX_SPEAKER_HEIGHT), 2)
                            train_sample = [
                                f"./neural_network/assets/train_sounds{filename}",
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
                                            speaker_height
                                            + round(
                                                uniform(
                                                    -MAX_PERTURBATION, MAX_PERTURBATION
                                                ),
                                                2,
                                            ),
                                        ),  # Speaker 1
                                        (
                                            round(uniform(0, x / 2), 2),
                                            round(uniform(y / 2, y), 2),
                                            speaker_height
                                            + round(
                                                uniform(
                                                    -MAX_PERTURBATION, MAX_PERTURBATION
                                                ),
                                                2,
                                            ),
                                        ),  # Speaker 2
                                        (
                                            round(uniform(x / 2, x), 2),
                                            round(uniform(y / 2, y), 2),
                                            speaker_height
                                            + round(
                                                uniform(
                                                    -MAX_PERTURBATION, MAX_PERTURBATION
                                                ),
                                                2,
                                            ),
                                        ),  # Speaker 3
                                        (
                                            round(uniform(x / 2, x), 2),
                                            round(uniform(0, y / 2), 2),
                                            speaker_height
                                            + round(
                                                uniform(
                                                    -MAX_PERTURBATION, MAX_PERTURBATION
                                                ),
                                                2,
                                            ),
                                        ),  # Speaker 4
                                        (
                                            randint(0, x),
                                            randint(0, y),
                                            speaker_height
                                            + round(
                                                uniform(
                                                    -MAX_PERTURBATION, MAX_PERTURBATION
                                                ),
                                                2,
                                            ),
                                        ),  # Subwoofer
                                    ],  # Speaker positions
                                    [
                                        choice(MATERIALS) for _ in range(6)
                                    ],  # 6 Materials
                                ],
                            ]
                            train_data.append(train_sample)

with open("./neural_network/train_data_input.csv", "w") as csvfile:
    # Create a writer object to write to the file
    csvwriter = writer(csvfile)

    # writing the data rows
    csvwriter.writerows(train_data)
