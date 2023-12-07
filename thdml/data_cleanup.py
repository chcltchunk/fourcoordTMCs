import pandas as pd
import numpy as np
from constants import column_list, sse_colum_list
from constants import S2_CUTOFF

# load raw data
raw_data_dir = "../data/"
df_hetero = pd.read_csv(raw_data_dir + "heteroleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")
df_homo = pd.read_csv(raw_data_dir + "homoleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")

# TODO(ralf): should we have charge in the final dataset?
# select relevant properties from dataset
# TODO(jonas): remove up until pd.to_csv and pd.load_csv instead
df = pd.concat([df_homo, df_hetero])

df = df[((df["geom.hs"] == "tetrahedral") | (df["geom.hs"] == "square planar")) | ((df["geom.ls"] == "tetrahedral") | (df["geom.ls"] == "square planar"))]
# TODO(ralf): I'd remove any detailed results for <S2> from the dataset before preparing it for the ML tasks?
df = df[((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)]

df = df[[*column_list, *sse_colum_list]]

df.to_csv(raw_data_dir + "thd_tmcs_geom_sse.csv")