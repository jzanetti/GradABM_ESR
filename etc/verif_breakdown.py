from collections import Counter
from glob import glob
from os.path import join
from pickle import load as pickle_load

from pandas import read_csv as pandas_read_csv

from process import AGE_INDEX, ETHNICITY_INDEX

swapped_ethnicity_dict = {value: key for key, value in ETHNICITY_INDEX.items()}
swapped_age_dict = {value: key for key, value in AGE_INDEX.items()}

gradabm_output_dir = "etc/tests/PHA_report_202403/predict"
obs_data = "etc/tests/papers/ehtnicity_breakdown_2019.csv"

obs_data = pandas_read_csv(obs_data)
obs_data["0-14"] = obs_data["<1"] + obs_data["1-4"] + obs_data["5-14"]
obs_data.drop(["<1", "1-4", "5-14"], axis=1, inplace=True)
total_age_counts = obs_data.iloc[:, 1:].sum(axis=0)
# Calculate the percentage for each value
obs_data_age_dict = total_age_counts / total_age_counts.sum()
obs_data_age_dict = obs_data_age_dict.to_dict()

obs_data["total_ethnicity"] = obs_data.sum(axis=1)
obs_data["precentage_ethnicity"] = (
    obs_data["total_ethnicity"] / obs_data["total_ethnicity"].sum()
)
obs_data_ethnicity_dict = dict(
    zip(obs_data["Unnamed: 0"], obs_data["precentage_ethnicity"])
)

age_mapping = {
    "0-14": ["0-10", "11-20"],
    "15-24": ["11-20", "21-30"],
    "25-44": ["21-30", "31-40"],
    "45-64": ["41-50", "51-60"],
    "65+": ["61-999"],
}

all_outputs = glob(join(gradabm_output_dir + "/*.pickle"))

all_data = {"ethnicity": [], "age": [], "vaccine": []}

for proc_file in all_outputs:
    proc_data = pickle_load(open(proc_file, "rb"))
    all_indices = proc_data["output"]["stages"]["all_indices"]
    total_timesteps = len(all_indices)

    all_ethnicity = []
    all_age = []
    all_vaccine = []

    for proc_t in range(total_timesteps):
        proc_indice = list(all_indices[proc_t])
        proc_ethnicity = [
            proc_data["output"]["agents"]["ethnicity"][i] for i in proc_indice
        ]
        proc_age = [proc_data["output"]["agents"]["age"][i] for i in proc_indice]
        proc_vaccine = [
            proc_data["output"]["agents"]["vaccine"][i] for i in proc_indice
        ]

        all_ethnicity.append(proc_ethnicity)
        all_age.append(proc_age)
        all_vaccine.append(proc_vaccine)

    all_data["ethnicity"].append(
        [item for sublist in all_ethnicity for item in sublist]
    )
    all_data["age"].append([item for sublist in all_age for item in sublist])
    all_data["vaccine"].append([item for sublist in all_vaccine for item in sublist])


occurrences_age = []
for proc_data in all_data["age"]:
    proc_occurence = Counter(proc_data)

    model_data_age_dict_tmp = {}
    for key, value in proc_occurence.items():
        model_data_age_dict_tmp[f"{swapped_age_dict[key]}"] = value

    model_data_age_dict = {}
    for proc_age_mapping in age_mapping:
        model_data_age_dict[proc_age_mapping] = 0
        for proc_key in age_mapping[proc_age_mapping]:
            model_data_age_dict[proc_age_mapping] += model_data_age_dict_tmp[proc_key]

    # Calculate the total sum of values
    total = sum(model_data_age_dict.values())

    # Convert values to percentages
    model_data_age_dict = {
        key: (value / total) for key, value in model_data_age_dict.items()
    }

    occurrences_age.append(model_data_age_dict)


occurrences_ethnicity = []
for proc_data in all_data["ethnicity"]:
    proc_occurence = Counter(proc_data)

    model_data_ethnicity_dict = {}
    for key, value in proc_occurence.items():
        model_data_ethnicity_dict[f"{swapped_ethnicity_dict[key]}"] = value

    model_data_ethnicity_dict["Other"] = (
        model_data_ethnicity_dict["European"] + model_data_ethnicity_dict["MELAA"]
    )

    model_data_ethnicity_dict = {
        key: value
        for key, value in model_data_ethnicity_dict.items()
        if key not in ["European", "MELAA"]
    }

    # Calculate the total sum of values
    total = sum(model_data_ethnicity_dict.values())

    # Convert values to percentages
    model_data_ethnicity_dict = {
        key: (value / total) for key, value in model_data_ethnicity_dict.items()
    }

    occurrences_ethnicity.append(model_data_ethnicity_dict)


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Create lists for each category
categories = ["Maori", "Pacific", "Asian", "Other"]
values = {category: [] for category in categories}

for entry in occurrences_ethnicity:
    for category in categories:
        values[category].append(entry[category])

# Create scatter plot
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    if i == 0:
        ax.scatter(
            [category] * len(values[category]),
            values[category],
            color="grey",
            alpha=0.3,
            label="Simulation",
        )
    else:
        ax.scatter(
            [category] * len(values[category]),
            values[category],
            color="grey",
            alpha=0.3,
        )

for category, value in obs_data_ethnicity_dict.items():
    if category == "Other":
        ax.scatter(
            category, value, marker="x", s=150, label="Truth", color="k"
        )  # s is the size of the marker
    else:
        ax.scatter(
            category, value, marker="x", s=150, color="k"
        )  # s is the size of the marker
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
ax.legend()
plt.xlabel("Ethnicity group")
plt.title("Percentage of Measles Infections by Ethnicity Group")
plt.savefig("ethnicity_breakdown.png", bbox_inches="tight")
plt.close()


# Create lists for each category
categories = list(model_data_age_dict.keys())
values = {category: [] for category in categories}

for entry in occurrences_age:
    for category in categories:
        values[category].append(entry[category])

# Create scatter plot
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    if i == 0:
        ax.scatter(
            [category] * len(values[category]),
            values[category],
            color="grey",
            alpha=0.3,
            label="Simulation",
        )
    else:
        ax.scatter(
            [category] * len(values[category]),
            values[category],
            color="grey",
            alpha=0.3,
        )

for category, value in obs_data_age_dict.items():
    if category == "0-14":
        ax.scatter(
            category, value, marker="x", s=150, label="Truth", color="k"
        )  # s is the size of the marker
    else:
        ax.scatter(
            category, value, marker="x", s=150, color="k"
        )  # s is the size of the marker
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
ax.legend()
plt.xlabel("Age group")
plt.title("Percentage of Measles Infections by Age Group")
plt.savefig("age_breakdown.png", bbox_inches="tight")
plt.close()
