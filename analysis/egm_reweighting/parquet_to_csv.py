#import xgboost
import pandas as pd
import glob
import re

# Load ten data events
file_path = "/vols/cms/pfk18/icenet_files/processed/"
files = glob.glob(f"{file_path}/*renamed*")
for f in files:
    print(f" --> Converting parquet to csv: {f}")
    events = pd.read_parquet(f)
    f_out = re.sub("parquet", "csv", f.split("/")[-1])
    events.to_csv(f"samples/{f_out}")
