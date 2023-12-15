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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def manage_skeewed_dataset(classes, features, samp_type="both"):
    #print(Counter(geometries))
    over = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
    if samp_type == "both":
        steps = [('o', over), ('u', under)]
    elif samp_type == "under":
        steps = [('u', under)]
    elif samp_type == "over":
        steps = [('o', over)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(features, classes)
    return y, X
    
