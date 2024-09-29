from torch import device as torch_device

# ---------------------------------
# Index control
# ---------------------------------
LOC_INDEX = {
    "household": 0,
    "supermarket": 1,
    "company": 2,
    "school": 3,
    "travel": 4,
    "restaurant": 5,
    "pharmacy": 6,
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

GENDER_INDEX = {"male": 0, "female": 1}

ETHNICITY_INDEX = {"European": 0, "Maori": 1, "Pacific": 2, "Asian": 3, "MELAA": 4}

VACCINE_INDEX = {"yes": 1, "no": 0}


# ---------------------------------
# Device control
# ---------------------------------
DEVICE = torch_device(f"cpu")  # torch_device(f"cuda:0") # cpu
