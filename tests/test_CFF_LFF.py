import numpy as np
from fcTMCml.constants import EPS
from fcTMCml.featurizer.CFF import CrystalFieldFeatures


def test_CFF_d_orbital_occupation():
    cff = CrystalFieldFeatures("cr", 3, 4)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [1, 1, 1, 0, 0])
    cff = CrystalFieldFeatures("mn", 3, 5)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [1, 1, 1, 1, 0])
    cff = CrystalFieldFeatures("mn", 2, 6)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [1, 1, 1, 1, 1])
    cff = CrystalFieldFeatures("fe", 3, 4)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 1, 1, 1, 0])
    cff = CrystalFieldFeatures("mn", 2, 2)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 1, 0, 0])
    cff = CrystalFieldFeatures("fe", 2, 1)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 2, 0, 0])
    cff = CrystalFieldFeatures("co", 2, 2)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 2, 1, 0])
    cff = CrystalFieldFeatures("ni", 2, 1)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 2, 2, 0])
    cff = CrystalFieldFeatures("ni", 2, 3)
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 2, 1, 1])
    cff = CrystalFieldFeatures("ni", 2, [3, 1])
    cff.occupy_d_orbitals()
    assert np.all(cff.occ[0] == [2, 2, 2, 1, 1])
    assert np.all(cff.occ[1] == [2, 2, 2, 2, 0])


def test_CFF_energy_features():
    cff = CrystalFieldFeatures("cr", 3, 4)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[0], cff.geometry_diff_of_quanta_values["tetrahedral"]) - (-3.56)) <= EPS
    cff = CrystalFieldFeatures("ni", 2, 3)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[1], cff.geometry_diff_of_quanta_values["square planar"]) - (-14.56)) <= EPS
    cff = CrystalFieldFeatures("mn", 3, 5)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[1], cff.geometry_diff_of_quanta_values["square planar"]) - (-12.28)) <= EPS
    cff = CrystalFieldFeatures("cr", 3, 2)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[0], cff.geometry_diff_of_quanta_values["tetrahedral"]) - (-8.01)) <= EPS
    cff = CrystalFieldFeatures("fe", 2, 1)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[0], cff.geometry_diff_of_quanta_values["tetrahedral"]) - (-7.12)) <= EPS
    cff = CrystalFieldFeatures("ni", 2, 1)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[1], cff.geometry_diff_of_quanta_values["square planar"]) - (-24.56)) <= EPS
    cff = CrystalFieldFeatures("cr", 2, 1)
    cff.occupy_d_orbitals()
    assert abs(cff.calculate_enes(cff.occ[1], cff.geometry_diff_of_quanta_values["square planar"]) - (-20.56)) <= EPS


def test_energy_diff():
    cff = CrystalFieldFeatures("cr", 2, [1, 3])
    assert abs(cff.get_energy_diff() - 9.02) <= EPS
    cff = CrystalFieldFeatures("cr", 2, [1, 3], geometryA="tetrahedral", geometryB="tetrahedral")
    assert abs(cff.get_energy_diff() + 4.449999) <= EPS
    cff = CrystalFieldFeatures("cr", 2, [1, 2], geometryA="tetrahedral", geometryB="tetrahedral")
    assert abs(cff.get_energy_diff() + 2.67) <= EPS
    cff = CrystalFieldFeatures("fe", 2, [1, 2], geometryA="tetrahedral", geometryB="tetrahedral")
    assert abs(cff.get_energy_diff() - 1.78) <= EPS
    cff = CrystalFieldFeatures("fe", 2, [1, 2], geometryA="square planar", geometryB="tetrahedral")
    assert abs(cff.get_energy_diff() + 20.22) <= EPS
    cff = CrystalFieldFeatures("fe", 2, [1, 2], geometryA="square planar", geometryB="square planar")
    assert abs(cff.get_energy_diff() + 4.28) <= EPS
