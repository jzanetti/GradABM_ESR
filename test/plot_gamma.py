import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

# Generate x values
x = np.linspace(0.1, 5, 100)

# Compute y values using the gamma function
y = gamma.pdf(x, a=1.5, scale=0.5)

# Create the plot
plt.plot(x, y)
plt.title("Gamma Function")
plt.xlabel("x")
plt.ylabel("Gamma(x)")

# Display the plot

# Display the plot
plt.savefig("test.png")
plt.close()
