import matplotlib.pyplot as plt
import numpy as np


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate x values
x = np.linspace(-100, -10, 100)

# Compute y values using the sigmoid function
y = sigmoid(x)

y0 = (y - y.min()) / (y.max() - y.min())


# Create the plot
plt.plot(x, y, label="orig")
plt.plot(x, y0, label="scaled")
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.legend()
plt.ylabel("sigmoid(x)")

# Display the plot
plt.savefig("test.png")
plt.close()
