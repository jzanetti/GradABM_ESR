import random

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import gamma

# Generate x values
x = np.linspace(0, 7, 100)

# Compute y values using the gamma function
y1 = gamma.pdf(x, a=2.41, scale=30.0)
y2 = gamma.pdf(x, a=2.41, scale=0.5)

# y = 10.0 * (y1 / max(y1))

# Create the plot
# plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="a=2.41; b=0.5")
# plt.title("Gamma Function")
plt.xlabel("Days after infection")
plt.ylabel("Infectiousness scaling factor")
# plt.legend()
# Display the plot

# Display the plot
plt.savefig("test.png")

# output = {"target": y}
# output = pandas.DataFrame.from_dict(output)
# output.to_csv("data/exp1/targets_test2.csv", index=False)
plt.close()
