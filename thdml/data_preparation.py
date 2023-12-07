import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load raw data
raw_data_dir = "../data/"
df_hetero = pd.read_csv(raw_data_dir + "heteroleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")
df_homo = pd.read_csv(raw_data_dir + "homoleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")

# define properties we'd like to keep throughout the whole cleanup
column_list = ['metal', 'ox', 'ligstr', 'complex.size']  # , 'charge'
sse_colum_list = ['geom.ls', 'geom.hs', 'ls.spin', 'hs.spin', 'b3lyp.energy.ls (Ha)', 'b3lyp.energy.hs (Ha)', 'b3lyp.sse (kcal/mol)']


# TODO(ralf): should we have charge in the final dataset?
# select relevant properties from dataset
# TODO(jonas): remove up until pd.to_csv and pd.load_csv instead
df = pd.concat([df_homo, df_hetero])
df_homo = df[[*column_list, *sse_colum_list,
              's2_is.ls', 's2_is.hs', 's2_expect.ls', 's2_expect.hs']]

df = df[((df["geom.hs"] == "tetrahedral") | (df["geom.hs"] == "square planar")) | ((df["geom.ls"] == "tetrahedral") | (df["geom.ls"] == "square planar"))]
# df = df[(np.abs(df["b3lyp.sse (kcal/mol)"]) < 110)]

# TODO(jonas): remove everything above and replace by pd.load_csv
# df = pd.load_csv("thd_geom_sse.csv")
df[[*column_list, *sse_colum_list]].to_csv("thd_geom_sse.csv")

###########################
# prepare classifier data #
###########################

count_tetrahedral = np.count_nonzero(df["geom.hs"] == "tetrahedral")
count_tetrahedral += np.count_nonzero(df["geom.ls"] == "tetrahedral")

count_square_planar = np.count_nonzero(df["geom.ls"] == "square planar")
count_square_planar += np.count_nonzero(df["geom.hs"] == "square planar")

print("count tetrahedral complexes: ", count_tetrahedral)
print("count square planar complexes: ", count_square_planar)

# in DOI: 10.1039/c7sc01247k cutoff is <= 1 (gives 242 less datapoints)
S2_CUTOFF = 1.5

# create dataset for classifier
df_classifier = pd.concat([df[(df["geom.hs"] == "tetrahedral")
                              & ((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)][[*column_list, 'hs.spin', 'geom.hs']],
                           df[(df["geom.ls"] == "tetrahedral")
                              & ((np.abs(df["s2_is.ls"] - df["s2_expect.ls"])) <= S2_CUTOFF)][[*column_list, 'ls.spin', 'geom.ls']],
                           df[(df["geom.hs"] == "square planar")
                              & ((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)][[*column_list, 'hs.spin', 'geom.hs']],
                           df[(df["geom.ls"] == "square planar")
                              & ((np.abs(df["s2_is.ls"] - df["s2_expect.ls"])) <= S2_CUTOFF)][[*column_list, 'ls.spin', 'geom.ls']]])


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
                       & (df["geom.ls"] == "tetrahedral")
                       & ((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)][[*column_list, *sse_colum_list]]

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
