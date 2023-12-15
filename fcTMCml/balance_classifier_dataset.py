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

targets = np.load(feature_target_dir + "classifier_targets.npy")

mcdl53_classifier_features = np.load(feature_target_dir + "MCDL53_classifier.npy")

mcdl53_cff_classifier_features = np.load(feature_target_dir + "MCDL53_cff_classifier.npy")

rac300_classifier_features = np.load(feature_target_dir + "RAC_classifier.npy")

rac300_cff_classifier_features = np.load(feature_target_dir + "RAC_cff_classifier.npy")

print("inital count square planars / total: ", np.count_nonzero(targets), "/", mcdl53_classifier_features.shape[0])
X, y = create_balanced_dataset(mcdl53_classifier_features, targets=targets)
print("square planar / total: ", np.count_nonzero(y), "/", X.shape[0])
X, y = create_balanced_dataset(mcdl53_cff_classifier_features, targets=targets)
print("square planar / total: ", np.count_nonzero(y), "/", X.shape[0])
X, y = create_balanced_dataset(rac300_classifier_features, targets=targets)
print("square planar / total: ", np.count_nonzero(y), "/", X.shape[0])
X, y = create_balanced_dataset(rac300_cff_classifier_features, targets=targets)
print("square planar / total: ", np.count_nonzero(y), "/", X.shape[0])
