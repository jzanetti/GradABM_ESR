import random

import pandas

num_agents = 100
agents = {"id": [], "age_group": []}
for i in range(num_agents):
    agents["id"].append(i)
    agents["age_group"].append(random.randint(0, 8))

df = pandas.DataFrame(agents)

df.to_csv("data/agents1.csv", index=False)

num_interactions = 500
interaction_grah_cfg = {"source": [], "target": [], "venue": []}

for i in range(num_interactions):
    src = random.randint(0, num_agents - 1)
    dest = random.randint(0, num_agents - 1)
    ve = random.randint(0, 1)

    interaction_grah_cfg["source"].append(src)
    interaction_grah_cfg["target"].append(dest)
    interaction_grah_cfg["venue"].append(ve)

df = pandas.DataFrame(interaction_grah_cfg)
df = df.drop_duplicates()
df.to_csv("data/interaction_graph_cfg1.csv", index=False)
