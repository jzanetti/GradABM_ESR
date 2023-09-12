import pickle

import matplotlib.pyplot as plt
import numpy

data_path = "/tmp/manukau_measles_2019_vac1/model/member_0/output_info.p"
data = pickle.load(open(data_path, "rb"))


proc_data = numpy.array(data["param_values_list"][-1][0, :, -1].cpu().detach().numpy())

sf = 100.0 / proc_data.max()

proc_data *= sf


def enlarge_differences(values, scaling_factor):
    new_values = [values[0]]

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        enlarged_diff = diff * scaling_factor
        new_value = new_values[-1] + enlarged_diff
        new_values.append(new_value)

    return new_values


_, ax = plt.subplots()

proc_data = enlarge_differences(proc_data, 300)
proc_data = numpy.round(proc_data, 4)
x = range(len(proc_data))
plt.plot(x, proc_data, "k")

start_timestep = 25
end_timestep = 51
xtick_positions = list(range(start_timestep, end_timestep))
tick_labels = [str(tick) for tick in xtick_positions]
ax.set_xticks(x[::3])
ax.set_xticklabels(tick_labels[::3])


plt.xlabel("Weeks")
plt.ylabel("Infectiousness Scaling Factor (%)")
plt.title("Infectiousness Scaling Factor \n Measles Outbreak 2019 (Week 26 - Week 51)")
plt.tight_layout()
plt.savefig("test.png", bbox_inches="tight")
plt.close()
