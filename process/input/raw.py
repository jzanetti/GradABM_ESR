from copy import deepcopy
from math import ceil as math_ceil
from os.path import join
from re import match as re_match

from numpy import inf, nan
from pandas import DataFrame, concat, melt, merge, read_csv, read_excel, to_numeric

from input import RAW_DATA, REGION_CODES, REGION_NAMES_CONVERSIONS


def create_population(workdir: str):
    """Read population

    Args:
        population_path (str): Population data path
    """
    data = read_excel(RAW_DATA["total_population"], header=6)

    data = data.rename(columns={"Area": "area", "Unnamed: 2": "population"})

    data = data.drop("Unnamed: 1", axis=1)

    # Drop the last row
    data = data.drop(data.index[-1])

    data = data.astype(int)

    data = data[data["area"] > 10000]

    data.to_csv(join(workdir, "total_population.csv"))

    return data


def create_geography_hierarchy(workdir: str):
    """Create geography

    Args:
        workdir (str): _description_
    """

    def _map_codes2(code: str) -> list:
        """Create a mapping function

        Args:
            code (str): Regional code to be mapped

        Returns:
            list: The list contains north and south island
        """
        for key, values in REGION_NAMES_CONVERSIONS.items():
            if code == key:
                return values
        return None

    data = read_csv(RAW_DATA["geography_hierarchy"])

    data = data[["REGC2023_code", "SA32023_code", "SA32023_name", "SA22018_code"]]

    data = data[~data["REGC2023_code"].isin(REGION_CODES["Others"])]

    data["REGC2023_name"] = data["REGC2023_code"].map(_map_codes2)

    data = data.rename(
        columns={
            "REGC2023_name": "region",
            "SA32023_code": "super_area",
            "SA22018_code": "area",
            "SA32023_name": "super_area_name",
        }
    ).drop_duplicates()

    data = data[["region", "super_area", "area", "super_area_name"]]

    data = data[~data["area"].duplicated(keep=False)]

    data.to_csv(
        join(workdir, "geography_hierarchy.csv"),
        index=False,
    )

    return data


def create_geography_location_super_area(
    workdir: str, geography_hierarchy_data: DataFrame
):
    data = read_csv(RAW_DATA["geography_location"])

    data = data[["SA22018_V1_00", "LATITUDE", "LONGITUDE"]]

    data = data.rename(
        columns={
            "SA22018_V1_00": "area",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    data = merge(data, geography_hierarchy_data, on="area", how="inner")

    data = data.groupby("super_area")[["latitude", "longitude"]].mean().reset_index()

    data.to_csv(
        join(workdir, "geography_location_super_area.csv"),
        index=False,
    )

    return data


def create_geography_location_area(workdir: str):
    """Write area location data

    Args:
        workdir (str): Working directory
        area_location_cfg (dict): Area location configuration
    """
    data = read_csv(RAW_DATA["geography_location"])

    data = data[["SA22018_V1_00", "LATITUDE", "LONGITUDE"]]

    data = data.rename(
        columns={
            "SA22018_V1_00": "area",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    data.to_csv(
        join(workdir, "geography_location_area.csv"),
        index=False,
    )


def create_socialeconomic(workdir: str, geography_hierarchy_data: DataFrame):
    """Write area area_socialeconomic_index data

    Args:
        workdir (str): Working directory
        area_socialeconomic_index_cfg (dict): Area_socialeconomic_index configuration
        geography_hierarchy_definition (DataFrame or None): Geography hierarchy definition
    """
    data = read_csv(RAW_DATA["socialeconomics"])[
        ["SA22018_code", "SA2_average_NZDep2018"]
    ]

    data = data.rename(
        columns={
            "SA22018_code": "area",
            "SA2_average_NZDep2018": "socioeconomic_centile",
        }
    )

    # get hierarchy defination data
    geog_hierarchy = geography_hierarchy_data[["super_area", "area"]]

    data = merge(data, geog_hierarchy, on="area")

    data.to_csv(
        join(workdir, "socialeconomics.csv"),
        index=False,
    )


def create_geography_name_super_area(workdir: str) -> dict:
    """Write super area names

    Args:
        workdir (str): Working directory
        use_sa3_as_super_area (bool): Use SA3 as super area, otherwise we will use regions
        geography_hierarchy_definition_cfg (dict or None): Geography hierarchy definition configuration
    """

    data = {"super_area": [], "city": []}

    data = read_csv(RAW_DATA["geography_hierarchy"])
    data = data[["SA32023_code", "SA32023_name"]]
    data = data.rename(columns={"SA32023_code": "super_area", "SA32023_name": "city"})
    data = data.drop_duplicates()
    data.to_csv(join(workdir, "geography_name_super_area.csv"), index=False)


def create_age(workdir: str, total_population_data: DataFrame):
    def _find_range(number, ranges):
        for age_range in ranges:
            start, end = map(int, age_range.split("-"))
            if start <= number <= end:
                return age_range
        return None

    df = read_excel(RAW_DATA["population_by_age"], header=2)

    df.columns = df.columns.str.strip()

    df = df[
        [
            "Region and Age",
            "0-4 Years",
            "5-9 Years",
            "10-14 Years",
            "15-19 Years",
            "20-24 Years",
            "25-29 Years",
            "30-34 Years",
            "35-39 Years",
            "40-44 Years",
            "45-49 Years",
            "50-54 Years",
            "55-59 Years",
            "60-64 Years",
            "65-69 Years",
            "70-74 Years",
            "75-79 Years",
            "80-84 Years",
            "85-89 Years",
            "90 Years and over",
        ]
    ]

    df = df.drop(df.index[-1])

    df["Region and Age"] = df["Region and Age"].str.strip()

    df = df[~df["Region and Age"].isin(["NZRC", "NIRC", "SIRC"])]

    df["Region and Age"] = df["Region and Age"].astype(int)

    df = df[df["Region and Age"] > 10000]

    df = df.set_index("Region and Age")

    df.columns = [str(name).replace(" Years", "") for name in df]
    df = df.rename(columns={"90 and over": "90-100"})

    new_df = DataFrame(columns=["Region"] + list(range(0, 101)))

    for cur_age in list(new_df.columns):
        if cur_age == "Region":
            new_df["Region"] = df.index
        else:
            age_range = _find_range(cur_age, list(df.columns))
            age_split = age_range.split("-")
            start_age = int(age_split[0])
            end_age = int(age_split[1])
            age_length = end_age - start_age + 1
            new_df[cur_age] = (df[age_range] / age_length).values

    new_df = new_df.applymap(math_ceil)

    new_df = new_df.rename(columns={"Region": "output_area"})

    all_ages = range(101)
    for index, row in new_df.iterrows():
        total = sum(row[col] for col in all_ages)
        new_df.at[index, "total"] = total

    total_population_data = total_population_data.rename(
        columns={"area": "output_area"}
    )
    df_after_ratio = new_df.merge(total_population_data, on="output_area")
    df_after_ratio["ratio"] = df_after_ratio["population"] / df_after_ratio["total"]

    for col in all_ages:
        df_after_ratio[col] = df_after_ratio[col] / df_after_ratio["ratio"]

    df_after_ratio.replace([inf, -inf], nan, inplace=True)
    df_after_ratio.dropna(inplace=True)

    df_after_ratio = df_after_ratio.round().astype(int)

    new_df = df_after_ratio.drop(["total", "population", "ratio"], axis=1)

    new_df.to_csv(join(workdir, "population_by_age.csv"), index=False)


def create_ethnicity_and_age(workdir: str, total_population_data: DataFrame):
    dfs = {}

    for proc_age_key in RAW_DATA["population_by_age_by_ethnicity"]:
        df = read_excel(
            RAW_DATA["population_by_age_by_ethnicity"][proc_age_key], header=4
        )
        df = df.drop([0, 1]).drop(df.tail(3).index)
        df = df.drop("Unnamed: 1", axis=1)
        df.columns = df.columns.str.strip()

        df = df.rename(
            columns={
                "Ethnic group": "output_area",
                "Pacific Peoples": "Pacific",
                "Middle Eastern/Latin American/African": "MELAA",
            }
        )

        df = (
            df.apply(to_numeric, errors="coerce").dropna().astype(int)
        )  # convert str ot others to NaN, and drop them and convert the rests to int

        df["total"] = (
            df["European"] + df["Maori"] + df["Pacific"] + df["Asian"] + df["MELAA"]
        )

        dfs[proc_age_key] = df

    df_ratio = concat(list(dfs.values()))
    df_ratio = df_ratio.groupby("output_area").sum().reset_index()
    total_population_data = total_population_data.rename(
        columns={"area": "output_area"}
    )
    df_ratio = df_ratio.merge(total_population_data, on="output_area")
    df_ratio["ratio"] = df_ratio["population"] / df_ratio["total"]
    df_ratio = df_ratio.drop(
        ["European", "Maori", "Pacific", "Asian", "MELAA", "total", "population"],
        axis=1,
    )

    dfs_after_ratio = {}
    for proc_age in dfs:
        df = dfs[proc_age]

        df = df.merge(df_ratio, on="output_area")
        for race_key in ["European", "Maori", "Pacific", "Asian", "MELAA", "total"]:
            df[race_key] = df[race_key] * df["ratio"]
        df = df.drop(["ratio", "total"], axis=1)
        # df = df.astype(int)
        # df = df.apply(math_ceil).astype(int)
        df = df.round().astype(int)
        dfs_after_ratio[proc_age] = df

    dfs = dfs_after_ratio

    dfs_output = []
    for proc_age in dfs:
        dfs_output.append(
            melt(
                dfs[proc_age],
                id_vars=["output_area"],
                value_vars=[
                    "European",
                    "Maori",
                    "Pacific",
                    "Asian",
                    "MELAA",
                ],
                var_name="ethnicity",
                value_name=proc_age,
            )
        )

    # Assuming 'dataframes' is a list containing your DataFrames
    combined_df = merge(dfs_output[0], dfs_output[1], on=["output_area", "ethnicity"])
    for i in range(2, len(dfs_output)):
        combined_df = merge(combined_df, dfs_output[i], on=["output_area", "ethnicity"])

    combined_df.to_csv(
        combined_df.to_csv(join(workdir, "ethnicity_and_age.csv"), index=False),
        index=False,
    )


def create_female_ratio(workdir: str):
    """Write gender_profile_female_ratio

    Args:
        workdir (str): Working directory
        gender_profile_female_ratio_cfg (dict): gender_profile_female_ratio configuration
    """

    df = read_excel(RAW_DATA["population_by_age_by_gender"], header=3)

    df = df.rename(
        columns={
            "Male": "Male (15)",
            "Female": "Female (15)",
            "Male.1": "Male (40)",
            "Female.1": "Female (40)",
            "Male.2": "Male (65)",
            "Female.2": "Female (65)",
            "Male.3": "Male (90)",
            "Female.3": "Female (90)",
            "Sex": "output_area",
        }
    )

    df = df.drop("Unnamed: 1", axis=1)

    df = df.drop([0, 1, 2]).drop(df.tail(3).index).astype(int)

    df = df[df["output_area"] > 10000]

    for age in ["15", "40", "65", "90"]:
        df[age] = df[f"Female ({age})"] / (df[f"Male ({age})"] + df[f"Female ({age})"])

    df = df[["output_area", "15", "40", "65", "90"]]

    df = df.dropna()

    df.to_csv(join(workdir, "population_by_gender.csv"), index=False)


def read_leed(
    leed_path: str, anzsic_code: DataFrame, if_rate: bool = False
) -> DataFrame:
    """Read NZ stats LEED data

    Args:
        leed_path (str): leed path to be processed
        anzsic_code (Dataframe): ANZSIC codes
        if_rate (bool): if return male and female rate

    Returns:
        DataFrame: Leed dataset
    """
    df = read_excel(leed_path)
    industrial_row = df.iloc[0].fillna(method="ffill")

    if anzsic_code is not None:
        for i, row in enumerate(industrial_row):
            row = row.strip()

            if row in ["Industry", "Total people"]:
                continue

            code = anzsic_code[anzsic_code["Description"] == row]["Anzsic06"].values[0]
            industrial_row[i] = code

    # x = anzsic_code.set_index("Description")
    sec_row = df.iloc[1].fillna(method="ffill")
    titles = industrial_row + "," + sec_row
    titles[
        "Number of Employees by Industry, Age Group, Sex, and Region (derived from 2018 Census)"
    ] = "Area"
    titles["Unnamed: 1"] = "Age"
    titles["Unnamed: 2"] = "tmp"

    df = df.iloc[3:]
    df = df.drop(df.index[-1:])
    df = df.rename(columns=titles)
    df = df.drop("tmp", axis=1)
    df["Area"] = df["Area"].fillna(method="ffill")
    # return df.rename(columns=lambda x: x.strip())

    df["Area"] = df["Area"].replace(
        "Manawatu-Wanganui Region", "Manawatu-Whanganui Region"
    )

    if anzsic_code is not None:
        character_indices = set(
            [
                col.split(",")[0][0]
                for col in df.columns
                if col
                not in ["Area", "Age", "Total people,Male", "Total people, Female"]
            ]
        )

        # Iterate over the unique character indices to sum the corresponding columns
        for char_index in character_indices:
            subset_cols_male = [
                col
                for col in df.columns
                if col.startswith(char_index)
                and col.endswith("Male")
                and col
                not in ["Area", "Age", "Total people,Male", "Total people,Female"]
            ]
            subset_cols_female = [
                col
                for col in df.columns
                if col.startswith(char_index)
                and col.endswith("Female")
                and col
                not in ["Area", "Age", "Total people,Male", "Total people,Female"]
            ]
            summed_col_male = f"{char_index},Male"
            summed_col_female = f"{char_index},Female"
            df[summed_col_male] = df[subset_cols_male].sum(axis=1)
            df[summed_col_female] = df[subset_cols_female].sum(axis=1)
            df = df.drop(subset_cols_male + subset_cols_female, axis=1)

    df["Area"] = df["Area"].str.replace(" Region", "")

    if not if_rate:
        return df

    industrial_columns = [
        x
        for x in list(df.columns)
        if x not in ["Area", "Age", "Total people,Male", "Total people,Female"]
    ]

    df = df.groupby("Area")[industrial_columns].sum()

    df_rate = deepcopy(df)

    # Calculate percentages
    for column in df.columns:
        group = column.split(",")[0]
        total = df[[f"{group},Male", f"{group},Female"]].sum(
            axis=1
        )  # Calculate the total for the group

        total.replace(0, nan, inplace=True)
        df_rate[column] = df[column] / total

    return df_rate


def read_anzsic_code(anzsic06_code_path: str) -> DataFrame:
    """Read ANZSIC code

    Args:
        anzsic06_code_path (str): ANZSIC data path

    Returns:
        DataFrame: code in a dataframe
    """
    anzsic_code = read_csv(anzsic06_code_path)

    for _, row in anzsic_code.iterrows():
        row["Description"] = " ".join(row["Description"].split()[1:])

    return anzsic_code


def write_employees(
    workdir: str,
    employees_cfg: dict,
    pop: DataFrame or None = None,
    use_sa3_as_super_area: bool = False,
):
    """Write the number of employees by gender for different area

    Args:
        workdir (str): Working directory
        employees_cfg (dict): Configuration
        use_sa3_as_super_area (bool): If apply SA3 as super area, otherwise using regions
    """

    def _rename_column(column_name):
        # Define a regular expression pattern to match the column names
        pattern = r"([a-zA-Z]+) ([a-zA-Z]+)"
        matches = re_match(pattern, column_name)
        if matches:
            gender = matches.group(1)[0].lower()
            category = matches.group(2)
            return f"{gender} {category}"
        return column_name

    data_path = get_raw_data(
        workdir,
        employees_cfg,
        "employees",
        "group/company",
        force=True,
    )

    # Read Leed rate for region
    data_leed_rate = read_leed(
        data_path["deps"]["leed"],
        read_anzsic_code(data_path["deps"]["anzsic06_code"]),
        if_rate=True,
    )

    # Read employees for different SA2
    data = read_csv(data_path["raw"])[["anzsic06", "Area", "ec_count"]]

    data = data[
        data["anzsic06"].isin(
            list(set([col.split(",")[0] for col in data_leed_rate.columns]))
        )
    ]

    data_sa2 = data[data["Area"].str.startswith("A")]

    data_sa2 = data_sa2.rename(columns={"Area": "area"})

    data_sa2["area"] = data_sa2["area"].str[1:].astype(int)

    # Read geography hierarchy
    geography_hierarchy_definition = read_csv(
        data_path["deps"]["geography_hierarchy_definition"]
    )

    # data_sa2 = data_sa2.merge(
    #    geography_hierarchy_definition[["area", "super_area", "region"]], on="area", how="left"
    # )

    if use_sa3_as_super_area:
        data_sa2 = data_sa2.merge(
            geography_hierarchy_definition[["area", "region"]], on="area", how="left"
        )
        data_sa2 = data_sa2.dropna()
    else:
        data_sa2 = data_sa2.merge(
            geography_hierarchy_definition[["area", "super_area"]],
            on="area",
            how="left",
        )
        data_sa2 = data_sa2.dropna()
        data_sa2["super_area"] = data_sa2["super_area"].astype(int)
        data_sa2 = data_sa2.rename(columns={"super_area": "region"})
        data_sa2["region"] = data_sa2["region"].map(REGION_NAMES_CONVERSIONS)

    data_leed_rate = data_leed_rate.reset_index()

    data_leed_rate = data_leed_rate.rename(columns={"Area": "region"})
    data_sa2 = data_sa2.merge(data_leed_rate, on="region", how="left")

    industrial_codes = []
    industrial_codes_with_genders = []
    for proc_item in list(data_sa2.columns):
        if proc_item.endswith("Male"):
            proc_code = proc_item.split(",")[0]
            industrial_codes.append(proc_code)
            industrial_codes_with_genders.extend(
                [f"{proc_code},Male", f"{proc_code},Female"]
            )

    # Create new columns 'male' and 'female' based on 'anzsic06' prefix
    for category in industrial_codes:
        male_col = f"{category},Male"
        female_col = f"{category},Female"
        data_sa2.loc[data_sa2["anzsic06"] == category, "Male"] = (
            data_sa2[male_col] * data_sa2["ec_count"]
        )
        data_sa2.loc[data_sa2["anzsic06"] == category, "Female"] = (
            data_sa2[female_col] * data_sa2["ec_count"]
        )

    df = data_sa2.drop(columns=industrial_codes_with_genders)

    if pop is not None:
        total_people_target = int(
            pop["population"].sum()
            * FIXED_DATA["group"]["company"]["employees"]["employment_rate"]
        )
        total_people_current = df["ec_count"].sum()
        people_factor = total_people_target / total_people_current

        df["Male"] = df["Male"] * people_factor
        df["Female"] = df["Female"] * people_factor
        df["ec_count"] = df["ec_count"] * people_factor

    df["Male"] = df["Male"].astype("int")
    df["Female"] = df["Female"].astype("int")
    df["ec_count"] = df["ec_count"].astype("int")

    df_pivot = pivot_table(
        df, index="area", columns="anzsic06", values=["Male", "Female"]
    )

    df_pivot.columns = [f"{col[0]} {col[1]}" for col in df_pivot.columns]

    df_pivot = df_pivot.fillna(0.0)
    df_pivot = df_pivot.astype(int)

    df_pivot = (
        df_pivot.rename(columns=_rename_column)
        .reset_index()
        .rename(columns={"area": "oareas"})
    )

    df_pivot.to_csv(data_path["output"], index=False)

    return {"data": df_pivot, "output": data_path["output"]}
