import csv

# These values are not the input values for the NN,
# but the inputs can be deduced from/with these values
# Path, master volume, mic_loc, room_measurements, speaker1_loc,
# speaker2_loc, speaker3_loc, speaker4_loc, speaker5_loc
test_case = [
    [
        "../assets/guiter_16k.wav",
        50,
        [
            (2, 2, 2),
            (10, 8, 6),
            [(1, 1, 1), (9, 9, 1), (4, 6, 2), (4, 10, 4), (5, 5, 5)],
            [
                "brickwork",
                "brickwork",
                "brickwork",
                "brickwork",
                "brickwork",
                "brickwork",
            ],
        ],
    ]
]

with open("train_data_input.csv", "w") as csvfile:
    # Create a writer object to write to the file
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(test_case)
