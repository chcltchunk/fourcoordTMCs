import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import column_list, sse_colum_list
from constants import SSE_CUTOFF


df = pd.read_csv("thd_geom_sse.csv")

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

print(df_classifier)
df_classifier.to_csv("thd_geom_classifier.csv")

###############################
# prepare SSE prediction data #
###############################

df_sse_prediction = df[(df["geom.hs"] == "tetrahedral")
                       & (df["geom.ls"] == "tetrahedral")][[*column_list, *sse_colum_list]]

# remove unreasonably high SSEs
SSE_CUTOFF = -110
df_sse_prediction = df_sse_prediction[df_sse_prediction["b3lyp.sse (kcal/mol)"] > SSE_CUTOFF]
print(df_sse_prediction)

# create numpy array with SSE for masking
sse_n = df_sse_prediction["b3lyp.sse (kcal/mol)"].to_numpy()

print("count SSEs: ", len(sse_n), " max SSE: ", np.max(sse_n), "; min SSE: ", np.min(sse_n))

# TODO(jonas): move to dataset analysis
# visualize SSE distribution in dataset
plt.hist(sse_n, bins=50)
plt.savefig("SSE distribution.png", dpi=300, bbox_inches="tight")

# count HS and LS preferences
print("HS preference", np.count_nonzero(sse_n >= 0))
print("LS preference", np.count_nonzero(sse_n < 0))