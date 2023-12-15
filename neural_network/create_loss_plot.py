import csv
import matplotlib.pyplot as plt

with open("./neural_network/loss_data.csv", "r") as file:
    reader = csv.reader(file)
    # Assuming there is only one row in the CSV file
    loss_data = next(reader)

# Convert the string values to floats
loss_data = [float(loss) for loss in loss_data]

print(len(loss_data))
plt.plot(range(1, len(loss_data) + 1), loss_data)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.show()
