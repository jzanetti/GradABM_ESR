from random import sample as random_sample

import numpy as np
import pandas as pd

from mesa import Agent, Model
from mesa.datacollection import DataCollector
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

all_people_id = list(df2.id.unique())

df2.loc[df2["id"] == 496996, "latitude"] = -39.0

import enum


class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2


class Person(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.unique_id = unique_id
        self.age = self.random.normalvariate(20, 40)
        self.pos = pos
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0

    def step(self):
        # Check for people within 3 meters
        if self.state != State.INFECTED:
            neighbors = self.model.grid.get_neighbors(
                self.pos, 0.0, include_center=True
            )
            for neighbor in neighbors:
                # if neighbor.unique_id == 496996:
                if neighbor.unique_id == self.unique_id:
                    continue
                if neighbor.unique_id == 496996:
                    x = 3
                if neighbor.state == State.SUSCEPTIBLE:
                    neighbor.state = State.INFECTED


class VirusModel(Model):
    def __init__(self, width, height):
        self.grid = ContinuousSpace(width, height, True)
        self.schedule = RandomActivation(self)

        # Create agents
        for i in all_people_id:
            if i == 496996:
                x = 3
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

        # make some agents infected at start
        person_agents = [
            agent for agent in self.schedule.agents if isinstance(agent, Person)
        ]

        # Randomly select 100 agents
        infected_agents = random_sample(person_agents, 10)

        # Label selected agents as infected
        for agent in infected_agents:
            agent.state = State.INFECTED

        self.datacollector = DataCollector(agent_reporters={"State": "state"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# Create model and run for 10 steps
model = VirusModel(10, 10)
for i in range(10):
    model.step()

agent_state = model.datacollector.get_agent_vars_dataframe()
agent_state.reset_index(inplace=True)


def get_column_data(model):
    """pivot the model dataframe to get states count at each step"""
    agent_state = model.datacollector.get_agent_vars_dataframe()
    df = pd.pivot_table(
        agent_state.reset_index(),
        index="Step",
        columns="State",
        aggfunc=np.size,
        fill_value=0,
    )
    labels = ["Susceptible", "Infected", "Removed"]
    df.columns = labels[: len(df.columns)]
    return df
