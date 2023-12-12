import numpy as np
from fcTMCml.constants import EPS
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
    

def test_get_ligand_denticity():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac"])
    assert mcdlf.get_ligand_denticity() == [4, 1, 1, 1, 1, 2]


def test_get_ligand_charges():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    assert mcdlf.get_ligand_charges() == [0, -1, -1, 0 , 0, -1, -2]

def test_get_coordinating_atom_numbers():
    mcdlf = MCDL46(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    print(mcdlf.get_coordinating_atom_numbers())
    assert mcdlf.get_coordinating_atom_numbers() == [7, 8, 17, 8]


def test_get_ligand_max_bond_order():
    pass
