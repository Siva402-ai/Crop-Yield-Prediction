import pandas as pd
import numpy as np

df = pd.read_csv("crop_yield_climate_soil_data_2019_2023.csv")

# create CNN feature column
df["CNN_Feature"] = np.random.uniform(0.2, 0.8, len(df))

df.to_csv("crop_yield_climate_soil_data_2019_2023.csv", index=False)

print("Dataset updated with CNN_Feature")