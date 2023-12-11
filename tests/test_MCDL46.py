import numpy as np
from fcTMCml.constants import EPS
from fcTMCml.featurizer.MCDL46 import MCDL46
from networkx import graph


def get_dummy_graph():
    g = graph.Graph()
    g.add_node(0, atomic_number=26)
    return g
    

def test_get_ligand_denticities():
    mcdlf = MCDL46(get_dummy_graph(), None, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac"])
    assert mcdlf.get_ligand_denticity() == [4, 1, 1, 1, 1, 2]


def test_get_ligand_charges():
    mcdlf = MCDL46(get_dummy_graph(), None, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    assert mcdlf.get_ligand_charges() == [0, -1, -1, 0 , 0, -1, -2]
