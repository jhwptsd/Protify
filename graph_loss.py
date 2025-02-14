import matplotlib.pyplot as plt

# Define the file path
file_path = "losses.txt"  # Change this to your actual file name

# Initialize lists
column1 = []
column2 = []

# Read the file and process each line
with open(file_path, "r") as file:
    for line in file:
        try:
            num1, num2 = map(float, line.strip().split("\t"))  # Convert to float
            column1.append(num1)
            column2.append(num2)
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")

# Data for plotting
s = [i for i in range(len(column1))]

fig, ax = plt.subplots()
ax.plot(s,column1)

ax.grid()
plt.savefig("loss.png")
plt.show()