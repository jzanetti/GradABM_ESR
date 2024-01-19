LOC_INDEX = {
    "household": 0,
    "supermarket": 1,
    "company": 2,
    "school": 3,
    "travel": 4,
    "restaurant": 5,
}

AGE_INDEX = {
    "0-10": 0,
    "11-20": 1,
    "21-30": 2,
    "31-40": 3,
    "41-50": 4,
    "51-60": 5,
    "61-999": 6,
}

ETHNICITY_INDEX = {"European": 0, "Maori": 1, "Pacific": 2, "Asian": 3, "MELAA": 4}

SEX_INDEX = {"m": 0, "f": 1}

TRAINING_ENS_MEMBERS = 1

RANDOM_ENSEMBLES = 2

# ---------------------------------
# Raw data information
# ---------------------------------
RAW_DATA = {
    "total_population": "data/inputs/raw/total_population-2018.xlsx",
    "geography_hierarchy": "data/inputs/raw/geography_hierarchy-2023.csv",
    "geography_location": "data/inputs/raw/geography_location-2018.csv",
    "socialeconomics": "data/inputs/raw/socialeconomics-2018.csv",
    "population_by_age_by_gender": "data/inputs/raw/population_age_gender-2022.csv",
    "population_by_age_by_ethnicity": {
        0: "data/inputs/raw/population_age_ethnicity_under_15years-2018.xlsx",
        15: "data/inputs/raw/population_age_ethnicity_between_15_and_29years-2018.xlsx",
        30: "data/inputs/raw/population_age_ethnicity_between_30_and_64years-2018.xlsx",
        65: "data/inputs/raw/population_age_ethnicity_between_over_65years-2018.xlsx",
    },
    "population_by_age": "data/inputs/raw/population_by_age-2022.xlsx",
    "employee_by_gender_by_sector": [
        "employee_by_area-2022.csv",
        "leed-2018.xlsx",
        "anzsic06_code.csv",
        "data/inputs/raw/geography_hierarchy-2023.csv",
    ],
}


REGION_CODES = {
    "North Island": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "South Island": [12, 13, 14, 15, 16, 17, 18],
    "Others": [99],
}
REGION_NAMES_CONVERSIONS = {
    1: "Northland",
    14: "Otago",
    3: "Waikato",
    99: "Area Outside",
    2: "Auckland",
    4: "Bay of Plenty",
    8: "Manawatu-Whanganui",
    15: "Southland",
    6: "Hawke's Bay",
    5: "Gisborne",
    7: "Taranaki",
    9: "Wellington",
    17: "Nelson",
    18: "Marlborough",
    13: "Canterbury",
    16: "Tasman",
    12: "West Coast",
}
