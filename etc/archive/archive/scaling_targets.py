import pandas as pd

all_agents = pd.read_csv("data/exp1/agents.csv")
all_targets = pd.read_csv("data/exp1/target_orig.csv")

total_agents = len(all_agents)

mortality_rate = 0.1
dead_people = total_agents * mortality_rate

orig = all_targets["target"].sum()
scaling_factor = dead_people / orig

all_targets["target"] = all_targets["target"] * scaling_factor

all_targets.to_csv("data/exp1/targets.csv", index=False)
