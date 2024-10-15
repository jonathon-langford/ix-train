#import xgboost
import pandas as pd
import glob
import re

# Load ten data events
file_path = "samples_eval"
files = glob.glob(f"{file_path}/*csv")
for f in files:
    print(f" --> Converting csv to parquet: {f}")
    events = pd.read_csv(f)
    f_out = re.sub("csv", "parquet", f.split("/")[-1])
    events.to_parquet(f"{file_path}/{f_out}")
