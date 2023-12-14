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
from fcTMCml.featurizer.CFF import CrystalFieldFeatures


class LigandFieldFeatures(CrystalFieldFeatures):
    def __init__(self, metal, ox, mult, ligand_list):
        super().__init__(metal, ox, mult)
        self.ligand_list = ligand_list

    def init_dictionaries(self):
        super().init_dictionaries()

        self.lambda_a = 2
        self.lambda_b = 2

        # !! water is factor 1 reference for these
        # see jorgensen
        self.cores_splitting_strengths = {"co" : {2: 9.3, 3 : 19.0},
                                          "ni" : {2 : 8.9},
                                          "cr" : {2 : 13.9, 3 : 17.4},
                                          "fe" : {2 : 10.4, 3 : 14.0},
                                          "mn" : {2 : 8.0, 3 : 12.8}  # Mn3+ is an educated guess... #TODO: adjust
                                          }
        ud = np.nan
        self.L_dict = {
            'acac': ud,
            'aceticacidbipyridine': ud,
            'acetonitrile': 1,
            'ammonia': 1,
            'bipy': 2,
            'chloride': ud,
            'cn': 2,
            'cyanate': 1,
            'cyanopyridine': ud,
            'fluoride': 1,
            'formaldehyde': ud,
            'furan': ud,
            'hydrogensulfide': ud,
            'hydroxyl': 1,
            'iodide': 0,
            'mebipy': ud,
            'mec': ud,
            'methylamine': ud,
            'misc': ud,
            'ncs': 0,
            'ox': 1,
            'phen': 2,
            'phenisc': ud,
            'phosphine': ud,
            'pisc': ud,
            'pyridine': 1,
            'scn': ud,
            'tbuc': ud,
            'thiopyridine': ud,
            'water': 1,
            'benzisc': ud,
            'carbonyl': 2,
            'en': 2,
            's2-': 0,
            'pph3': 2
        }
        # ligand field strength parameter
        # 10.1246/bcsj.61.693
        self.L_dict = {
            'acac': 10.0,
            'aceticacidbipyridine': ud,
            'acetonitrile': 23.5,
            'ammonia': 21.0,
            'bipy': 22.2,
            'chloride': 12.5,
            'cn': 32.1,
            'cyanate': 15.8,
            'cyanopyridine': 20.0,  # not verified
            'fluoride': 1,
            'formaldehyde': 15.0,  # not verified
            'furan': ud,
            'hydrogensulfide': ud,
            'hydroxyl': 1,
            'iodide': 9.0,
            'mebipy': ud,
            'mec': ud,
            'methylamine': 20.0,
            'misc': ud,
            'ncs': 13.0,
            'ox': 16.6,
            'phen': 22.0,
            'phenisc': ud,
            'phosphine': ud,
            'pisc': ud,
            'pyridine': 20.8,
            'scn': 16.3,
            'tbuc': ud,
            'thiopyridine': ud,
            'water': 16.5,
            'benzisc': 19.5,  # not verified
            'carbonyl': 35.0,
            'en': 21.4,
            's2-': 11.8,
            'pph3': 23.2
        }

    def lambda_param(self) -> float:
        L = [self.L_dict[i] for i in self.ligand_list]
        h_1 = np.heaviside(self.lambda_a - self.ox, 1)
        h_2 = np.heaviside(self.lambda_b - (np.sum(L)), 1)
        # use continuos representation
        metal_splitting = self.cores_splitting_strengths[self.metal][self.ox]
        h_1 = metal_splitting / 20  # 17.4
        h_2 = np.sum(L) / (35 * len(self.ligand_list))  # normalize by effect of 4 (thd/sqp) CO ligands and add 1 for upscaling
        # normalize by effect of 4 (thd/sqp) water ligands and add 1 for upscaling
        # metal based splitting is based on water as reference system
        return np.round(h_1 * h_2 + 1, 2)

    def calculate_enes(self, occ: np.array, dqs: np.array) -> float:
        dqs = dqs * self.lambda_param()
        return super().calculate_enes(occ, dqs)
