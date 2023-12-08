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
from fcTMCml.constants import EPS, deg_sqp, dqs_sqp, deg_thd, dqs_thd


class CrystalFieldFeatures():
    def __init__(self, metal, ox, mult, lig_list=None, EPS=1e-6, fac=0.621, lambda_a=2, lambda_b=4):
        self.factor = fac
        self.lambda_a = 2
        self.lambda_b = 2
        self.metal = metal.lower()
        self.ox = int(ox)
        self.mult = int(mult)
        self.lig_list = lig_list
        self.init_dictionaries()
        


    def init_dictionaries(self): 
        # TODO add back 
        # self.cores_d_conf = {"co" : [7, 6],
        #          "ni" : [8],
        #          "cr" : [4, 3],
        #          "fe" : [6, 5],
        #          "mn" : [5, 4]
        #             }
        self.cores_d_conf = {"co" : {2: 7, 3 : 6},
                 "ni" : {2 : 8},
                 "cr" : {2 : 4, 3 : 3},
                 "fe" : {2 : 6, 3 : 5},
                 "mn" : {2 : 5, 3 : 4}
                    }
        
        # define geometry based degeneracy patterns from and dq energy schemes
        self.geometry_degeneracies = {
            "square planar" : [2, 1, 1, 1],
            "tetrahedral" : [2, 3]
        }
        self.geometry_diff_of_quanta = {
            "tetrahedral" : np.array([-2.67, -2.67, 1.78, 1.78, 1.78]),
            "square planar" : np.array([-5.14, -5.14, -4.28, 2.28, 12.28])
        }





    def calculate_enes(self, occ, dqs):  
        conf_ene = np.dot(occ, dqs)
        return conf_ene


    def occupy_d_orbitals(self):
        single_es = self.mult-1 # for spin 1/2
        num_es = self.cores_d_conf[self.metal][self.ox]
        if num_es == 10:
            if single_es == 0:
                double_occ = 10
            else:
                raise "electron config and multiplicity do not match"
        double_occ = (num_es - single_es)//2
        occ = [0]*5
        self.mult
        for i in range(double_occ):
            occ[i] = 2
        for i in range(double_occ, double_occ+single_es):
            occ[i] = 1
        return occ    

    def get_energy_diff_sqp_thd(self):
        occ = self.occupy_d_orbitals()
        thd_ene = self.calculate_enes(occ, self.thd_dqs)
        sqp_ene = self.calculate_enes(occ, self.sqp_dqs)
        return thd_ene-sqp_ene

    def get_energy_diff_sqp_thd_and_geom_guess(self, dbloc=False):
        enediff = self.get_energy_diff_sqp_thd(dbloc)
        guess = 0 if enediff < 0 else 1 
        return enediff, guess
    
        


dbloc = CrystalFieldFeatures("cr", 3, 4, [])
assert np.all(dbloc.occupy_d_orbitals() == [1,1,1,0,0])
dbloc = CrystalFieldFeatures("mn", 3, 5, [])
assert np.all(dbloc.occupy_d_orbitals() == [1,1,1,1,0])
dbloc = CrystalFieldFeatures("mn", 2, 6, [])
assert np.all(dbloc.occupy_d_orbitals() == [1,1,1,1,1])
dbloc = CrystalFieldFeatures("fe", 3, 4, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,1,1,1,0])
dbloc = CrystalFieldFeatures("mn", 2, 2, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,2,1,0,0])
dbloc = CrystalFieldFeatures("fe", 2, 1, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,2,2,0,0])
dbloc = CrystalFieldFeatures("co", 2, 2, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,2,2,1,0])
dbloc = CrystalFieldFeatures("ni", 2, 1, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,2,2,2,0])
dbloc = CrystalFieldFeatures("ni", 2, 3, [])
assert np.all(dbloc.occupy_d_orbitals() == [2,2,2,1,1])


dbloc = CrystalFieldFeatures("cr", 3, 4, [])
assert np.all(dbloc.calculate_comb_factors([1,0,0,0,0], dbloc.thd_deg) == [1, 0])
assert np.all(dbloc.calculate_comb_factors([1,1,0,0,0], dbloc.thd_deg) == [2, 0])
assert np.all(dbloc.calculate_comb_factors([1,1,1,0,0], dbloc.thd_deg) == [2, 1])
assert np.all(dbloc.calculate_comb_factors([1,1,1,1,0], dbloc.thd_deg) == [2, 2])
assert np.all(dbloc.calculate_comb_factors([1,1,1,1,1], dbloc.thd_deg) == [2, 3])
assert np.all(dbloc.calculate_comb_factors([2,1,1,1,1], dbloc.thd_deg) == [1, 3])
assert np.all(dbloc.calculate_comb_factors([2,2,1,1,1], dbloc.thd_deg) == [0, 3])
assert np.all(dbloc.calculate_comb_factors([2,2,2,1,1], dbloc.thd_deg) == [0, 2])
assert np.all(dbloc.calculate_comb_factors([2,2,1,1,1], dbloc.sqp_deg) == [0, 1, 1, 1])

# assert dbloc.calculate_exchange_mimic([1,0,1,0]) == 1
# assert dbloc.calculate_exchange_mimic([2,1,0,0]) == 2
# assert (dbloc.calculate_exchange_mimic_relative_energies([2,1,2,0], dbloc.sqp_dqs) - 3.169543) <= EPS 
# assert dbloc.calculate_exchange_mimic_single_energy_level([2,1,2,0]) == 4*0.5
# assert dbloc.calculate_exchange_mimic_single_energy_level([2,3,2,0]) == 10*0.5






EPS = 1e-6
dbloc = CrystalFieldFeatures("cr", 3, 4, None)
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.thd_dqs) - (-3.56)) <= EPS 
dbloc = CrystalFieldFeatures("ni", 2, 3, ["test_dummy"])
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.sqp_dqs, False) - (-14.56)) <= EPS 
dbloc = CrystalFieldFeatures("mn", 3, 5, None)
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.sqp_dqs, False) - (-12.28)) <= EPS 
dbloc = CrystalFieldFeatures("cr", 3, 2, None)
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.thd_dqs) - (-8.01)) <= EPS 
dbloc = CrystalFieldFeatures("fe", 2, 1, [])
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.thd_dqs, False) - (-7.12)) <= EPS 
dbloc = CrystalFieldFeatures("ni", 2, 1, [])
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.sqp_dqs, False) - (-24.56)) <= EPS 
dbloc = CrystalFieldFeatures("cr", 2, 1, [])
occ = dbloc.occupy_d_orbitals()
assert abs(dbloc.calculate_enes(occ, dbloc.sqp_dqs, False) - (-20.56)) <= EPS 



def extend_racs(
    rac_set,
    per_rack_identifiers,
    extension_type='default'
):
    """
    works for identifiers of the format:
        metal_co_ox_2_spin_2_ligstr_ligand1_ligand2_ligand3_ligand4_*
    adds 9 additional elements to feature vector
    extension_type : str
        type of extension: 'default', 'phys_ediffs', 'phys_ediffs_only' 'dbloc_ediffs'
    """
    extensions = []
    for index, name in enumerate(per_rack_identifiers):
        split = name.split("_") 
        metal = str(split[1])
        ox = int(split[3]) 
        mult = int(split[5]) 
        lig_list = split[7:11]
        dents = []
        lig_charges = []
        for lig in lig_list:
            dent_info = molsimplify_ligand_dict[lig][2]
            dents += [1 if (type(dent_info) == str) else len(dent_info)]
            lig_charges += [int(molsimplify_ligand_dict[lig][5][0])]
        if extension_type == 'default':
            extensions += [[ox, mult, *dents]]
        elif extension_type == 'phys_ediffs':
            dbloc = CrystalFieldFeatures(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ox, ediff, guess, *lig_charges, *dents]]
        elif extension_type == 'phys_ediffs_only':
            dbloc = CrystalFieldFeatures(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ediff, guess]]
        elif extension_type == 'dbloc_ediffs':
            dbloc = CrystalFieldFeatures(metal, ox, mult, lig_list)
            occ = dbloc.occupy_d_orbitals()
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=True)
            extensions += [[ox, ediff, guess, *lig_charges, *dents]]
        else:
            raise ValueError("define a proper extension type: default, phys_ediffs, 'phys_ediffs_only', dbloc_ediffs")
    extensions = np.vstack(extensions)
    rac_set = rac_set.reshape(-1, np.prod(rac_set.shape[1:]))
    extended_racs = np.hstack((extensions, rac_set))
    return extended_racs


def extend_mcdl46(
    mcdl46_set,
    per_mcdl46_identifiers,
    extension_type='phys_ediffs'
):
    """
    works for identifiers of the format:
        metal_co_ox_2_spin_2_ligstr_ligand1_ligand2_ligand3_ligand4_*
    adds 9 additional elements to feature vector
    extension_type : str
        type of extension: 'phys_ediffs', 'dbloc_ediffs'
    """
    extensions = []
    for index, name in enumerate(per_mcdl46_identifiers):
        split = name.split("_") 
        metal = str(split[1])
        ox = int(split[3]) 
        mult = int(split[5]) 
        lig_list = split[7:11]
        dents = []
        lig_charges = []
        for lig in lig_list:
            dent_info = molsimplify_ligand_dict[lig][2]
            dents += [1 if (type(dent_info) == str) else len(dent_info)]
            lig_charges += [int(molsimplify_ligand_dict[lig][5][0])]
        if extension_type == 'phys_ediffs':
            dbloc = CrystalFieldFeatures(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ediff, guess]]
        elif extension_type == 'dbloc_ediffs':
            dbloc = CrystalFieldFeatures(metal, ox, mult, lig_list)
            occ = dbloc.occupy_d_orbitals()
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=True)
            extensions += [[ediff, guess]]
        else:
            raise ValueError("define a proper extension type: phys_ediffs, dbloc_ediffs")
    extensions = np.vstack(extensions)
    extended_racs = np.hstack((extensions, mcdl46_set.reshape(-1, np.prod(mcdl46_set.shape[1:]))))
    print(extended_racs.shape)
    return extended_racs

