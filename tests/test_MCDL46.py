import numpy as np
from fcTMCml.constants import EPS, test_resource_dir
from fcTMCml.featurizer.MCDL46 import MCDL46
from networkx import graph


def get_dummy_graph():
    g = graph.Graph()
    g.add_node(0, atomic_number=26)
    g.add_node(1, atomic_number=7)
    g.add_node(2, atomic_number=8)
    g.add_node(3, atomic_number=17)
    g.add_node(4, atomic_number=8)
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return g


def test_get_classifier_features():
    # TODO
    assert (1 - 1) < EPS


def test_get_regression_features():
    # TODO:
    assert (1 - 1) < EPS


def test_get_classifier_feature_names(self):
    pass


def test_get_regression_feature_names(self):
    pass


def test_get_truncated_graph(self, truncation: int = None):
    pass


def test_get_ligands_as_subgraph():
    pass


def test_get_electronegativity_diffs():
    pass


def test_get_electronegativity_features():
    pass


def test_get_coordinating_atom_numbers():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    print(mcdlf.get_coordinating_atom_numbers())
    assert mcdlf.get_coordinating_atom_numbers() == [7, 8, 17, 8]


def test_get_ligand_charges():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    assert mcdlf.get_ligand_charges() == [0, -1, -1, 0 , 0, -1, -2]


def test_get_ligand_denticity():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac"])
    assert mcdlf.get_ligand_denticity() == [4, 1, 1, 1, 1, 2]


def test_get_number_of_atoms():
    pass


def test_get_ligand_number_of_atoms(self) -> list:
    self.ligands_as_subgraph_n = self.get_ligands_as_subgraph()
    ligand_sizes_n = [ligand[1].number_of_nodes() for ligand in self.ligands_as_subgraph_n]
    return ligand_sizes_n


def test_get_ligand_max_bond_order():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    assert int(np.max(mcdlf.get_ligand_max_bond_order(test_resource_dir + "furan.mol"))) == 2
    assert int(np.max(mcdlf.get_ligand_max_bond_order(test_resource_dir + "water.xyz"))) == 1


def test_get_kier_index():
    pass


def test_get_all_ligands_atom_counts():
    pass
