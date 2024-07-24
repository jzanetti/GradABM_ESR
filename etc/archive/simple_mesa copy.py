import pandas as pd

from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

syspop_base = pd.read_parquet("/tmp/syspop_test/Auckland/syspop_base.parquet")
syspop_diary = pd.read_parquet("/tmp/syspop_test/Auckland/syspop_diaries.parquet")
syspop_address = pd.read_parquet("/tmp/syspop_test/Auckland/syspop_location.parquet")

syspop_diary = syspop_diary[syspop_diary["hour"] == "12"]

syspop_diary = syspop_diary.sample(10000, random_state=10)[["id", "location"]]


syspop_address = syspop_address[["name", "latitude", "longitude"]]
syspop_address = syspop_address.rename(columns={"name": "location"})

interactions = pd.merge(syspop_diary, syspop_diary, on="location")
interactions = interactions[interactions["id_x"] != interactions["id_y"]]
interactions = interactions[["id_x", "id_y", "location"]]

merged_df = pd.merge(
    interactions, syspop_address, on="location", how="left"
).drop_duplicates()

# Rename the columns to match the MESA model's input format
merged_df = merged_df.rename(
    columns={
        "id_x": "person_1",
        "id_y": "person_2",
        "location": "interaction_location",
        "latitude": "location_lat",
        "longitude": "location_lon",
    }
)

df1 = merged_df[["person_1", "person_2"]].copy()
df1.columns = ["id_x", "id_y"]

# Create df2 from merged_df
df2_person_1 = merged_df[["person_1", "location_lat", "location_lon"]].copy()
df2_person_1.columns = ["id", "latitude", "longitude"]

df2_person_2 = merged_df[["person_2", "location_lat", "location_lon"]].copy()
df2_person_2.columns = ["id", "latitude", "longitude"]

df2 = pd.concat([df2_person_1, df2_person_2]).drop_duplicates()

# Assuming the dataframes are df1 (interaction data) and df2 (coordinates data)
# df1 = pd.DataFrame({"id_x": [13, 14, 15], "id_y": [15, 16, 17]})
# df2 = pd.DataFrame(
#    {"id": [13, 14, 15, 16, 17], "lat": [1, 2, 3, 4, 5], "lon": [6, 7, 8, 9, 10]}
# )

all_people_id = list(df2.id.unique())


class Person(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.pos = pos
        self.infected = False

    def step(self):
        # Check for people within 3 meters
        neighbors = self.model.grid.get_neighbors(self.pos, 0.0001, include_center=True)
        for neighbor in neighbors:
            if neighbor.infected:
                self.infected = True
                break


class VirusModel(Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = ContinuousSpace(width, height, True)
        self.schedule = RandomActivation(self)

        # Create agents
        for i in all_people_id:
            a = Person(
                i,
                self,
                (
                    df2.loc[df2["id"] == i, "latitude"].values[0],
                    df2.loc[df2["id"] == i, "longitude"].values[0],
                ),
            )

            self.schedule.add(a)
            self.grid.place_agent(a, a.pos)

        x = 3

    def step(self):
        self.schedule.step()


# Create model and run for 10 steps
model = VirusModel(100, 10, 10)
for i in range(10):
    model.step()
