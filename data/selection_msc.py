import numpy as np
import pandas as pd

with open("selection.csv", "r") as file:
    names = [line.rstrip() for line in file]


thd_sse = pd.read_csv("thd_sse_prediction.csv", index_col=0)
print(thd_sse)

new_sse = pd.DataFrame()

for i, row in thd_sse.iterrows():
    curr_name = f'metal_{row["metal"]}_ox_{int(row["ox"])}_spin_{int(row["ls.spin"])}_ligstr_{row["ligstr"]}'
    if curr_name in names:
        new_sse = new_sse._append(row, ignore_index=True)

new_sse = new_sse.reindex()

print(new_sse)

new_sse.to_csv("thd_sse_prediction_selection.csv", sep=",", index=True)
