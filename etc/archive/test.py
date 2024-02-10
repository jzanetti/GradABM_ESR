import random

import matplotlib.pyplot as plt
import numpy as np

x = [
    4000,
    2000,
    1500,
    1300,
    1200,
    1100,
    1000,
    900,
    850,
    800,
    750,
    700,
    685,
    650,
    630,
    600,
    595,
    590,
    583,
    581,
    580,
    579,
    580.5,
]
new_x = np.linspace(0, len(x) - 1, 100)
interpolated_values = np.interp(new_x, range(len(x)), x)

plt.plot(interpolated_values, "k")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.tight_layout()
plt.title("Loss, Manukau DHB \n Measles Outbreak 2019 (Week 26 - Week 51)")
plt.savefig("test.png", bbox_inches="tight")
plt.close()
