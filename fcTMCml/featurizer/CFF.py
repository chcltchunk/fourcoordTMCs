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
# implements ligand field features
# =============================================================================
# Imports
# =============================================================================
import numpy as np


class CrystalFieldFeatures():
    def __init__(self, metal, ox, mult):
        self.metal = metal.lower()
        self.ox = int(ox)
        self.mult = int(mult)
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
        self.geometry_diff_of_quanta = {
            "tetrahedral" : np.array([-2.67, -2.67, 1.78, 1.78, 1.78]),
            "square planar" : np.array([-5.14, -5.14, -4.28, 2.28, 12.28])
        }

    def calculate_enes(self, occ, dqs):
        conf_ene = np.dot(occ, dqs)
        return conf_ene

    def occupy_d_orbitals(self):
        single_es = self.mult - 1  # for spin 1/2
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
        return occ

    def get_energy_diff_sqp_thd(self):
        occ = self.occupy_d_orbitals()
        thd_ene = self.calculate_enes(occ, self.geometry_diff_of_quanta["tetrahedral"])
        sqp_ene = self.calculate_enes(occ, self.geometry_diff_of_quanta["square planar"])
        return thd_ene - sqp_ene

    def get_energy_diff_sqp_thd_and_geom_guess(self):
        enediff = self.get_energy_diff_sqp_thd()
        guess = 0 if enediff < 0 else 1
        return enediff, guess
