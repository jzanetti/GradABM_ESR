import geopandas as gpd
from pandas import read_csv

sa2_path = "/home/zhangs/Github/JUNE_NZ_data/geography/statistical-area-2-2018-centroid-inside.csv"
dhb_path = "/home/zhangs/Github/JUNE_NZ_data/geography/district-health-board-2015.csv"

sa2 = read_csv(sa2_path)
dhb = read_csv(dhb_path)

sa2["geometry"] = gpd.GeoSeries.from_wkt(sa2["WKT"])
dhb["geometry"] = gpd.GeoSeries.from_wkt(dhb["WKT"])

sa2_gdf = gpd.GeoDataFrame(sa2, geometry="geometry")
dhb_gdf = gpd.GeoDataFrame(dhb, geometry="geometry")
sa2_with_dhb = gpd.sjoin(sa2_gdf, dhb_gdf, how="left", op="within")

dhb_and_sa2 = sa2_with_dhb[["SA22018_V1_00", "DHB2015_Code", "DHB2015_Name"]]

dhb_and_sa2.columns = ["SA2", "DHB_code", "DHB_name"]

dhb_and_sa2.to_parquet("data/dhb_and_sa2.parquet")
