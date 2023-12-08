import pandas as pd
import numpy as np
from constants import column_list, sse_colum_list
from constants import S2_CUTOFF, SSE_CUTOFF

######################################################################
# DISCLAIMER                                                         #
# -------------------------------------------------------------------#                              
# This script is for reference only!                                 #
# It provides insight on how calculations were presorted             #
# by applying an S2 cutoff and eliminating unreasonably low SSEs.    #
######################################################################


# load raw data
raw_data_dir = "../data/"
df_hetero = pd.read_csv(raw_data_dir + "heteroleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")
df_homo = pd.read_csv(raw_data_dir + "homoleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", sep=";")

# combine the two raw datasets
df = pd.concat([df_homo, df_hetero])

# remove everything where neither LS or HS is tetrahedral or square planar
df = df[((df["geom.hs"] == "tetrahedral") | (df["geom.hs"] == "square planar")) | ((df["geom.ls"] == "tetrahedral") | (df["geom.ls"] == "square planar"))]

# remove calculation with an <S2> deviation larger S2_CUTOFF 
df = df[((np.abs(df["s2_is.hs"] - df["s2_expect.hs"])) <= S2_CUTOFF)]
# remove calculation with an SSE lower than SSE_CUTOFF
df = df[df["b3lyp.energy.ls (Ha)"] < SSE_CUTOFF]

# mask only the relevant property columns from the dataset
df = df[[*column_list, *sse_colum_list]]

# store dataset that is supplied with SI
df.reset_index(drop=True).to_csv(raw_data_dir + "thd_tmcs_geom_sse.csv")