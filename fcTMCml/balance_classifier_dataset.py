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

np.random.seed(0)


def create_balanced_dataset(features, targets, sampling_type="over"):
    over = SMOTE(sampling_strategy=0.7, random_state=42, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
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

feature_dict = load_features(feature_target_dir, classification_in_subdir, "classification")
targets, mcdl53_features, mcdl53_cff_features, rac300_features, rac300_cff_features = feature_dict.values()

print("inital count square planars / total: ", np.count_nonzero(targets), "/", mcdl53_features.shape[0])
X, y1 = create_balanced_dataset(mcdl53_features, targets=targets)
np.save(feature_target_dir + classification_out_subdir + "MCDL53_classifier.npy", X)
print("square planar / total: ", np.count_nonzero(y1), "/", X.shape[0])
X, y2 = create_balanced_dataset(mcdl53_cff_features, targets=targets)
np.save(feature_target_dir + classification_out_subdir + "MCDL53_cff_classifier.npy", X)
print("square planar / total: ", np.count_nonzero(y2), "/", X.shape[0])
X, y3 = create_balanced_dataset(rac300_features, targets=targets)
np.save(feature_target_dir + classification_out_subdir + "RAC_classifier.npy", X)
print("square planar / total: ", np.count_nonzero(y3), "/", X.shape[0])
X, y4 = create_balanced_dataset(rac300_cff_features, targets=targets)
np.save(feature_target_dir + classification_out_subdir + "RAC_cff_classifier.npy", X)
print("square planar / total: ", np.count_nonzero(y4), "/", X.shape[0])
assert list(y1) == list(y2) == list(y3) == list(y4)
np.save(feature_target_dir + classification_out_subdir + "classifier_targets.npy", y1)
