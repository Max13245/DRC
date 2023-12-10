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
            (2, 2, 2),
            (10, 16, 8),
            [(3, 10, 1), (2, 5, 3), (4, 11, 2), (9, 13, 2), (10, 12, 6)],
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
