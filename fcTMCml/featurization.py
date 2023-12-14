import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from featurizer.RAC import RAC
from featurizer.MCDL53 import MCDL53
from featurizer.LFF import LigandFieldFeatures
from featurizer.CFF import CrystalFieldFeatures

from fcTMCml.constants import raw_data_dir

"""
TODO:
- construct MCDL53 features
- construct MCDL53+CFF features
- construct RAC300 features
- construct RAC300+CFF features
- scale all features
- construct MCDL53+LFF features
- construct RAC300+LFF features
- scale features

"""
