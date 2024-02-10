import matplotlib.pyplot as plt
import numpy as np


# Define the sigmoid function
def _continuous_approximation(x):
    return 1 / (1 + np.exp(-10 * (x + 1)))


# Generate x values
x = np.linspace(-10, 10, 100)

# Compute y values using the sigmoid function
y = _continuous_approximation(x)


# Create the plot
plt.plot(x, y, label="orig")
plt.title("Custm Function")
plt.xlabel("x")
plt.legend()
plt.ylabel("y")

# Display the plot
plt.savefig("test.png")
plt.close()
