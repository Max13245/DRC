import csv

# These values are not the input values for the NN,
# but the inputs can be deduced from/with these values
# Path, master volume, mic_loc, room_measurements, speaker1_loc,
# speaker2_loc, speaker3_loc, speaker4_loc, speaker5_loc
test_case = [
    [
        "./neural_network/assets/guitar_16k.wav",
        50,
        [
            (5, 5, 5),
            (10, 16, 8),
            [(10, 0, 8), (2, 10, 3), (4, 0, 2), (3, 2, 0), (3, 0, 7)],
            [
                "carpet_cotton",
                "carpet_cotton",
                "carpet_cotton",
                "carpet_cotton",
                "carpet_cotton",
                "carpet_cotton",
            ],
        ],
    ]
]

with open("./neural_network/train_data_input.csv", "w") as csvfile:
    # Create a writer object to write to the file
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(test_case)
