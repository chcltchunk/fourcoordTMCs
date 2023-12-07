import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# in DOI: 10.1039/c7sc01247k cutoff is <= 1 (gives 242 less datapoints)
S2_CUTOFF = 1.5
# SSE lower than -110 kcal/mol is not reasonable for this dataset
SSE_CUTOFF = -110

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

df = df[((df["geom.hs"] == "tetrahedral") | (df["geom.hs"] == "square planar")) | ((df["geom.ls"] == "tetrahedral") | (df["geom.ls"] == "square planar"))]
# TODO(ralf): I'd remove any detailed results for <S2> from the dataset before preparing it for the ML tasks?
df = df[((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)]

df = df[[*column_list, *sse_colum_list]]

df.to_csv(raw_data_dir + "thd_tmcs_geom_sse.csv")