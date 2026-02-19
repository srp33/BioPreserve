import pandas as pd

df = pd.read_csv('/data/gold/gse_20194_62944/unadjusted.csv')
# select the "ESR1", "meta_source", and "meta_er_status" columns
df = df[['ESR1', 'meta_source', 'meta_er_status']]
# drop rows with missing values
df = df.dropna()

# save to "simple.csv"
df.to_csv('/data/gold/gse_20194_62944/simple.csv', index=False)