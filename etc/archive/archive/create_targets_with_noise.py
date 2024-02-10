import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

all_targets = pd.read_csv("data/exp1/targets_test2.csv")

lower_percentage = -10.0
upper_percentage = 10.0

all_targets_list = list(all_targets["target"])
noisy1_list = []
noisy2_list = []
for value in all_targets_list:
    noise1 = value * np.random.uniform(lower_percentage / 100, upper_percentage / 100)
    noise2 = (
        value * np.random.uniform(lower_percentage * 3 / 100, upper_percentage * 5 / 100)
        - value * 0.5
    )
    noisy_value1 = value + noise1
    noisy_value2 = value + noise2
    noisy1_list.append(noisy_value1)
    noisy2_list.append(noisy_value2)

data = {"x1": noisy1_list, "x2": noisy2_list}
df2 = pd.DataFrame(data)
df2.to_csv("data/exp1/targets_test2_with_noise.csv", index=False)
plt.plot(noisy1_list, "r")
plt.plot(noisy2_list, "r--")
plt.plot(all_targets_list, "b")

plt.savefig("test.png")
plt.close()
