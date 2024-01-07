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
# implements crystal field features
# =============================================================================
# Imports
# =============================================================================
import numpy as np


class CrystalFieldFeatures():
    def __init__(self, metal, ox, mult, geometryA="tetrahedral", geometryB="square planar"):
        self.metal = metal.lower()
        self.ox = int(ox)
        self.mult = (int(mult[0]), int(mult[1])) if type(mult) is list else (int(mult), int(mult))
        self.geometryA = geometryA
        self.geometryB = geometryB
        self.init_dictionaries()

    def init_dictionaries(self):
        self.cores_d_conf = {"co" : {2: 7, 3 : 6},
                             "ni" : {2 : 8},
                             "cr" : {2 : 4, 3 : 3},
                             "fe" : {2 : 6, 3 : 5},
                             "mn" : {2 : 5, 3 : 4}
                             }

        # define geometry based degeneracy patterns
        self.geometry_degeneracies = {
            "square planar" : [2, 1, 1, 1],
            "tetrahedral" : [2, 3]
        }
        # define geometry based differential of quanta energy schemes
        self.geometry_diff_of_quanta_values = {
            "tetrahedral" : np.array([-2.67, -2.67, 1.78, 1.78, 1.78]),
            "square planar" : np.array([-5.14, -5.14, -4.28, 2.28, 12.28])
        }

    def calculate_enes(self, occ, dqs):
        conf_ene = np.dot(occ, dqs)
        return np.round(conf_ene, 2)

    def occupy_d_orbitals(self):
        self.occ = []
        for i, mult in enumerate(self.mult):
            single_es = mult - 1  # for spin 1/2
            num_es = self.cores_d_conf[self.metal][self.ox]
            if num_es == 10:
                if single_es == 0:
                    double_occ = 10
                else:
                    raise "electron config and multiplicity do not match"
            double_occ = (num_es - single_es) // 2
            occ = [0] * 5
            for i in range(double_occ):
                occ[i] = 2
            for i in range(double_occ, double_occ + single_es):
                occ[i] = 1
            self.occ += [occ]

    def get_energy_diff(self):
        """
        calculate energy difference between two geometries
        - used as continous CFF (regression)
        """
        self.occupy_d_orbitals()
        geomA_energy = self.calculate_enes(self.occ[0], self.geometry_diff_of_quanta_values[self.geometryA])
        geomB_energy = self.calculate_enes(self.occ[1], self.geometry_diff_of_quanta_values[self.geometryB])
        return geomA_energy - geomB_energy

    def get_geometry_guess(self):
        """
        returns 0 if geometryA is favored else 1
        - used as binary CFF (classification)
        """
        energy_diff = self.get_energy_diff()
        guess = 0 if energy_diff < 0 else 1
        return guess

    def get_classifier_features(self, additional_featurizer: list = []) -> list:
        feature_array = [self.get_energy_diff(), self.get_geometry_guess()]
        for featurizer in additional_featurizer:
            feature_array += featurizer.get_classifier_features()
        return feature_array

    def get_classifier_feature_names(self, additional_featurizer: list = []) -> list:
        feature_names = ['delE', 'GG']
        for featurizer in additional_featurizer:
            feature_names += featurizer.get_classifier_feature_names()
        return feature_names

    def get_regression_features(self, additional_featurizer: list = []) -> list:
        feature_array = [self.get_energy_diff()]
        for featurizer in additional_featurizer:
            feature_array += featurizer.get_classifier_features()
        return feature_array

    def get_regression_feature_names(self, additional_featurizer: list = []) -> list:
        feature_names = ['delE']
        for featurizer in additional_featurizer:
            feature_names += featurizer.get_classifier_feature_names()
        return feature_names
