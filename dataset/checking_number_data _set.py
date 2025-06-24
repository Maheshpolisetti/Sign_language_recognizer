import pandas as pd
import json


labels=["Hello","Bye","Thanks","Drink","Love"]
dfs=[]
for l in labels:
    df=pd.read_csv(f"seq_data_{l}.csv")
    dfs.append(df)

# combining all seq data 
total_data = pd.concat(dfs, ignore_index=True)
total_data.to_csv("seq_total_data.csv",index=False)

# checking total data set
df = pd.read_csv(f"seq_total_data.csv")
sequence = json.loads(df.iloc[0]["sequence"])
print(f"Frames: {len(sequence)}, Frame size: {len(sequence[0])},number of samples {len(df)}")
