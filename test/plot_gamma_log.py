import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

a = 1.5  # concentration parameter
b = 0.1  # rate parameter
total_t = 30  # total number of values

x = np.arange(total_t)  # range of values

gamma_dist = gamma(a, scale=1 / b)  # create a gamma distribution object

res = gamma_dist.logpdf(
    x
)  # calculate the log probability density function (PDF) and exponentiate it

res = np.exp(res)

plt.plot(res)
plt.savefig("test.png")
plt.close()
