import pandas as pd

mimic_df_pa = pd.read_parquet("data/mimiciii_patients_admissions.parquet")

# list column names
print(mimic_df_pa.columns)
