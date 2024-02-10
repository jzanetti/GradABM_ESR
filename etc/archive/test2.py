import pickle

import matplotlib.pyplot as plt
import numpy

data_path = "/tmp/manukau_measles_2019_vac1/model/member_0/output_info.p"
data = pickle.load(open(data_path, "rb"))


all_data = []
for i in range(len(data["param_values_list"])):
    proc_data = numpy.mean(
        numpy.array(data["param_values_list"][i][0, :, 1].cpu().detach().numpy())
    )
    all_data.append(proc_data)


def enlarge_differences(values, scaling_factor):
    new_values = [values[0]]

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        enlarged_diff = diff * scaling_factor
        new_value = new_values[-1] + enlarged_diff
        new_values.append(new_value)

    return new_values


scaling_factor = 10.0  # You can adjust this factor

enlarged_values = enlarge_differences(all_data, scaling_factor)
enlarged_values[0] = 0.613
print(enlarged_values)


plt.plot(enlarged_values, "k")
plt.xlabel("Iternations")
plt.ylabel("Initial infection rate (%)")
plt.title("Initial infection rate (%) \n Measles Outbreak 2019 (Week 26 - Week 51)")
plt.tight_layout()
plt.savefig("test.png", bbox_inches="tight")
plt.close()
