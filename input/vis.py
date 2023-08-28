from matplotlib.pyplot import axis, close, figure, pie, savefig, tight_layout, title

from input import AGE_INDEX, ETHNICITY_INDEX, SEX_INDEX


def agents_vis(agents, sa2):
    if sa2 is not None:
        agents_to_plot = agents[agents["area"].isin(sa2)]
    else:
        agents_to_plot = agents

    for column_name in ["age", "ethnicity", "sex"]:
        value_counts = agents_to_plot[column_name].value_counts() / len(agents_to_plot) * 100
        if column_name == "age":
            label_dict = {value: key for key, value in AGE_INDEX.items()}
            agents_to_plot["modified_age"] = agents_to_plot["age"].replace(
                {2: "2-5", 3: "2-5", 4: "2-5", 5: "2-5", 0: "0-1", 1: "0-1"}
            )
            label_dict["2-5"] = "21-60"
            label_dict["0-1"] = "0-20"
            label_dict[6] = "61 and above"
            value_counts = (
                agents_to_plot["modified_age"].value_counts() / len(agents_to_plot) * 100
            )

        elif column_name == "ethnicity":
            label_dict = {value: key for key, value in ETHNICITY_INDEX.items()}
        elif column_name == "sex":
            label_dict = {value: key for key, value in SEX_INDEX.items()}

        figure(figsize=(8, 8))
        pie(
            value_counts,
            labels=[label_dict.get(index, index) for index in value_counts.index],
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 16},
        )
        title(
            f"Percentage of people in each {column_name} group\n County Manukau DHB",
            fontsize=20,
        )
        axis("equal")  # Equal aspect ratio ensures that the pie is drawn as a circle.
        tight_layout()
        savefig(f"test_{column_name}.png", bbox_inches="tight")
        close()
