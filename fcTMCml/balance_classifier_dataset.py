#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright 2021, Jonas Oldenstaedt <joldenstaedt@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#
# =============================================================================

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import make_dir, load_features

classification_in_subdir = "classification_raw/"

classification_out_subdir = "classification_balanced/"

make_dir(feature_target_dir + classification_out_subdir)

np.random.seed(128)


def create_balanced_dataset(features, targets, sampling_type="over"):
    over = SMOTE(sampling_strategy=1, random_state=128, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=128)
    if sampling_type == "both":
        steps = [('o', over), ('u', under)]
    elif sampling_type == "under":
        steps = [('u', under)]
    elif sampling_type == "over":
        steps = [('o', over)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(features, targets)
    return X, y


###########################
# Geometry Classification #
###########################

targets, feature_dict, feature_names_dict, feature_groups_dict = load_features(feature_target_dir, classification_in_subdir, "classification", load_groups=True)
# mcdl53_features, mcdl53_cff_features, rac300_features, rac300_cff_features = feature_dict.values()

y_n = []
for run_ident in feature_dict:
    print("inital count square planars / total: ", np.count_nonzero(targets), "/", feature_dict[run_ident].shape[0])

    X, y = create_balanced_dataset(feature_dict[run_ident], targets=targets)
    y_n += [y]
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}.npy", X)
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}_names.npy", feature_names_dict[run_ident])
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}_groups.npy", feature_groups_dict[run_ident])
    print("square planar / total: ", np.count_nonzero(y), "/", X.shape[0])

assert list(y_n[0]) == list(y_n[1]) == list(y_n[2]) == list(y_n[3])
np.save(feature_target_dir + classification_out_subdir + "classification_targets.npy", y_n[0])
