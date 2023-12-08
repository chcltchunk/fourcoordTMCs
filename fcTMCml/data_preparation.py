import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import column_list, sse_colum_list

raw_data_dir = "../data/"
df = pd.read_csv(raw_data_dir + "thd_tmcs_geom_sse.csv")

###########################
# prepare classifier data #
###########################

count_tetrahedral = np.count_nonzero(df["geom.hs"] == "tetrahedral")
count_tetrahedral += np.count_nonzero(df["geom.ls"] == "tetrahedral")

count_square_planar = np.count_nonzero(df["geom.ls"] == "square planar")
count_square_planar += np.count_nonzero(df["geom.hs"] == "square planar")

print("count tetrahedral complexes: ", count_tetrahedral)
print("count square planar complexes: ", count_square_planar)

# create dataset for classifier
df_classifier = pd.concat([df[(df["geom.hs"] == "tetrahedral")][[*column_list, 'hs.spin', 'geom.hs']],
                           df[(df["geom.ls"] == "tetrahedral")][[*column_list, 'ls.spin', 'geom.ls']],
                           df[(df["geom.hs"] == "square planar")][[*column_list, 'hs.spin', 'geom.hs']],
                           df[(df["geom.ls"] == "square planar")][[*column_list, 'ls.spin', 'geom.ls']]])


# remove distinction between ls and hs (irrelevant for geometry prediction)
df_classifier["hs.spin"] = df_classifier["hs.spin"].fillna(df_classifier["ls.spin"])
df_classifier["geom.hs"] = df_classifier["geom.hs"].fillna(df_classifier["geom.ls"])
df_classifier = df_classifier.rename(columns={"hs.spin": "spin", "geom.hs": "geom"})
df_classifier = df_classifier[[*column_list, "spin", "geom"]]

df_classifier.reset_index(drop=True).to_csv(raw_data_dir + "thd_geom_classifier.csv")


# make sure there are no duplicates in the dataset
assert 0==np.count_nonzero(df.duplicated(["metal", "ox", "ligstr"]).to_numpy()))


###############################
# prepare SSE prediction data #
###############################

# prepare square planar dataset

df_sse_prediction = df[(df["geom.hs"] == "square planar")
                       & (df["geom.ls"] == "square planar")][[*column_list, *sse_colum_list]]

df_sse_prediction.reset_index(drop=True).to_csv(raw_data_dir + "sqp_sse_prediction.csv")

# prepare tetrahedral dataset

df_sse_prediction = df[(df["geom.hs"] == "tetrahedral")
                       & (df["geom.ls"] == "tetrahedral")][[*column_list, *sse_colum_list]]

df_sse_prediction.reset_index(drop=True).to_csv(raw_data_dir + "thd_sse_prediction.csv")

# create numpy array with SSE for masking
sse_n = df_sse_prediction["b3lyp.sse (kcal/mol)"].to_numpy()

print("count SSEs: ", len(sse_n), " max SSE: ", np.max(sse_n), "; min SSE: ", np.min(sse_n))

# TODO(jonas): move to dataset analysis
# visualize SSE distribution in dataset
plt.hist(sse_n, bins=50)
plt.savefig(raw_data_dir + "SSE distribution.png", dpi=300, bbox_inches="tight")

# count HS and LS preferences
print("HS preference", np.count_nonzero(sse_n >= 0))
print("LS preference", np.count_nonzero(sse_n < 0))