from os.path import exists

import pandas as pd
import tabula

tables = []
region_names = None
for week_num in range(54):
    pdf_path = f"etc/measlesReport/measlesReport2019_{week_num}.pdf"

    print(pdf_path)
    if not exists(pdf_path):
        continue

    all_tables = tabula.read_pdf(pdf_path, pages=2)  # [1].drop(0)

    for table_i in range(len(all_tables)):
        if week_num >= 35:
            all_tables[table_i].columns = all_tables[table_i].iloc[0]
            all_tables[table_i] = all_tables[table_i].drop(0)

        if "Cumulative" in all_tables[table_i].columns:
            proc_table = all_tables[table_i]
            if week_num in [33, 34]:
                proc_table = proc_table.drop(columns=proc_table.columns[-1:])
            if len(all_tables[table_i].columns) > 4:
                proc_table = proc_table.drop(columns=proc_table.columns[-3:])
            break

    if week_num < 33:
        proc_table.columns = ["Region", f"Week_{week_num}", "Sum"]
        proc_table = proc_table.drop(0)
        proc_table.drop(proc_table.index[-1], inplace=True)
        tables.append(proc_table[["Region", f"Week_{week_num}"]])
    else:
        proc_table = proc_table.dropna()
        try:
            proc_table = proc_table.drop(0)
        except KeyError:
            pass

        if week_num >= 35:
            proc_table.drop(proc_table.index[-1], inplace=True)

            if len(proc_table) == 21:
                proc_table = proc_table.drop(1)

            if week_num in [36, 50]:
                proc_table.columns = ["Region", f"Week_{week_num-1}", f"Week_{week_num}"]
            else:
                proc_table.columns = ["temp_x1", "temp_x2"]
                proc_table[["Region", f"Week_{week_num-1}"]] = proc_table["temp_x1"].str.extract(
                    r"^(.*\D)(\d+)$"
                )
                proc_table = proc_table.drop(columns=["temp_x1"])
                proc_table = proc_table.rename(columns={"temp_x2": f"Week_{week_num}"})
                proc_table = proc_table[["Region", f"Week_{week_num-1}", f"Week_{week_num}"]]
        else:
            # proc_table = proc_table.drop(columns=proc_table.columns[-1])
            proc_table.columns = ["Region", f"Week_{week_num-1}", f"Week_{week_num}"]
            proc_table.drop(proc_table.index[-1], inplace=True)

        if region_names is None:
            proc_table["Region"].replace(
                {
                    "Wa2ir4a-rMaapya": "Wairarapa",
                    "Nel3s1o-nM Mayarlborough": "Nelson Marlborough",
                    "West7 C-Jouanst": "West Coast",
                    "Can1te4r-Jbuunry": "Canterbury",
                    "Sou2th1 -CJuannterbury": "South Canterbury",
                    "Sou2th8e-Jrunn": "Southern",
                },
                inplace=True,
            )
            region_names = list(proc_table["Region"])
        else:
            proc_table["Region"] = region_names

        tables.append(proc_table)

from functools import reduce

merged_df = reduce(lambda left, right: pd.merge(left, right, on="Region", how="outer"), tables)
columns_to_remove = merged_df.filter(like="_y").columns
merged_df = merged_df.drop(columns=columns_to_remove)
merged_df.columns = [col.rstrip("_x") if col.endswith("_x") else col for col in merged_df.columns]
merged_df.to_parquet("measles_cases_2019.parquet")
"""
merged_df.set_index("Region", inplace=True)
row_as_list = merged_df.loc[merged_df["Region"] == "Counties Manukau"].iloc[0].tolist()[1:]
float_list = [float(x) for x in row_as_list]
import matplotlib.pyplot as plt

plt.plot(float_list)
plt.savefig("test.png")
plt.close()
"""
