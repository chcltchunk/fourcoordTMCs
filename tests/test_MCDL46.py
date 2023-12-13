import numpy as np
from networkx import graph

from fcTMCml.constants import EPS, test_resource_dir
from fcTMCml.featurizer.MCDL53 import MCDL53
from fcTMCml.tools import openbabel_available


def get_dummy_graph():
    g = graph.Graph()
    g.add_node(0, atomic_number=26)
    # N_2
    g.add_node(1, atomic_number=7)
    g.add_node(5, atomic_number=7)
    # HO^-
    g.add_node(2, atomic_number=8)
    g.add_node(6, atomic_number=1)
    # Cl
    g.add_node(3, atomic_number=17)
    # HO^-
    g.add_node(4, atomic_number=8)
    g.add_node(7, atomic_number=1)
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (4, 7)])
    return g


def test_get_classifier_features():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"],
                   multiplicity=2, input_file=test_resource_dir + "dummy.xyz")
    features = mcdlf.get_classifier_features()
    if openbabel_available():
        assert len(features) == 53
    else:
        assert len(features) == 52
    if openbabel_available():
        assert np.all(np.array(features) - np.array([26, 3, 5.76, 1.21, 1.61, 2, 1, 7, 8, 17, 8, 0, -1, -1,
                                                    0, 0, -1, -2, 4, 1, 1, 1, 1, 2, 1, 8, 2, 2, 1, 2, 1, 3.11, 3.11,
                                                    0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0]) < EPS)
    else:
        assert np.all(np.array(features) - np.array([26, 3, 5.76, 1.21, 1.61, 2, 1, 7, 8, 17, 8, 0, -1, -1,
                                                    0, 0, -1, -2, 4, 1, 1, 1, 1, 2, 1, 8, 2, 2, 1, 2, 3.11, 3.11,
                                                    0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0]) < EPS)


def test_get_regression_features():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"],
                   multiplicity=2, input_file=test_resource_dir + "dummy.xyz")
    features = mcdlf.get_regression_features()
    if openbabel_available():
        assert len(features) == 51
    else:
        assert len(features) == 50
    if openbabel_available():
        assert np.all(np.array(features) - np.array([26, 3, 5.76, 1.21, 1.61, 7, 8, 17, 8, 0, -1, -1,
                                                    0, 0, -1, -2, 4, 1, 1, 1, 1, 2, 1, 8, 2, 2, 1, 2, 1, 3.11, 3.11,
                                                    0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0]) < EPS)
    else:
        assert np.all(np.array(features) - np.array([26, 3, 5.76, 1.21, 1.61, 7, 8, 17, 8, 0, -1, -1,
                                                    0, 0, -1, -2, 4, 1, 1, 1, 1, 2, 1, 8, 2, 2, 1, 2, 3.11, 3.11,
                                                    0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0]) < EPS)



def test_get_classifier_feature_names():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"],
                   multiplicity=2, input_file=test_resource_dir + "dummy.xyz")
    features = mcdlf.get_classifier_features()
    names = mcdlf.get_classifier_feature_names()
    assert len(names) == len(features)


def test_get_regression_feature_names():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"],
                   multiplicity=2, input_file=test_resource_dir + "dummy.xyz")
    features = mcdlf.get_regression_features()
    names = mcdlf.get_regression_feature_names()
    assert len(names) == len(features)


def test_get_feature_dict():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"],
                   multiplicity=2, input_file=test_resource_dir + "dummy.xyz")
    features = mcdlf.get_regression_features()
    names = mcdlf.get_regression_feature_names()
    feature_dict = mcdlf.get_feature_dict(names, features)
    if openbabel_available():
        assert feature_dict == {'I(M)': 26, 'Ox': 3, 'sum($\\chi$)': 5.76, 'min($\\chi$)': 1.21, 'max($\\chi$)': 1.6099999999999999,
                                'CA': 8, 'LC': -2, 'LD': 1, '#A': 8, 'L#A': 2, 'max_LBO': 1, 'K': 3.11, 'TK': 3.11,
                                '#B': 0, '#C': 0, '#N': 2, '#O': 2, '#F': 0, '#P': 0, '#S': 0, '#Cl': 1, '#Br': 0, '#I': 0,
                                'T#B': 0, 'T#C': 0, 'T#N': 2, 'T#O': 2, 'T#F': 0, 'T#P': 0, 'T#S': 0, 'T#Cl': 1, 'T#Br': 0, 'T#I': 0}
    else:
        assert feature_dict == {'I(M)': 26, 'Ox': 3, 'sum($\\chi$)': 5.76, 'min($\\chi$)': 1.21, 'max($\\chi$)': 1.6099999999999999,
                                'CA': 8, 'LC': -2, 'LD': 1, '#A': 8, 'L#A': 2, 'K': 3.11, 'TK': 3.11,
                                '#B': 0, '#C': 0, '#N': 2, '#O': 2, '#F': 0, '#P': 0, '#S': 0, '#Cl': 1, '#Br': 0, '#I': 0,
                                'T#B': 0, 'T#C': 0, 'T#N': 2, 'T#O': 2, 'T#F': 0, 'T#P': 0, 'T#S': 0, 'T#Cl': 1, 'T#Br': 0, 'T#I': 0}


def test_get_truncated_graph():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    graph = mcdlf.get_truncated_graph(1)
    assert len(graph.nodes) == 5


def test_get_ligands_as_subgraph():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ligands_as_subgraph = mcdlf.get_ligands_as_subgraph()
    print(ligands_as_subgraph)
    for i, an in [[[1, 5], [7, 7]], [[2, 6], [8, 1]], [3, 17], [[4, 7], [8, 1]]]:
        g_ref = graph.Graph()
        if type(i) is list:
            g_ref.add_node(i[0], atomic_number=an[0])
            g_ref.add_node(i[1], atomic_number=an[1])
            ligand_index = i[0] - 1
            g_ref.add_edges_from([(i[0], i[1])])
        else:
            g_ref.add_node(i, atomic_number=an)
            ligand_index = i - 1
        print(g_ref.nodes, ligands_as_subgraph[0][1].nodes)
        print(g_ref.edges, ligands_as_subgraph[0][1].edges)
        assert ligands_as_subgraph[ligand_index][1].nodes == g_ref.nodes
        assert ligands_as_subgraph[ligand_index][1].edges == g_ref.edges


def test_get_ligands_as_subgraph_truncated():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ligands_as_subgraph = mcdlf.get_ligands_as_subgraph(truncation=1)
    print(ligands_as_subgraph)
    for i, an in [[1, 7], [2, 8], [3, 17], [4, 8]]:
        g_ref = graph.Graph()
        g_ref.add_node(i, atomic_number=an)
        assert ligands_as_subgraph[i - 1][1].nodes == g_ref.nodes
        assert ligands_as_subgraph[i - 1][1].edges == g_ref.edges


def test_get_electronegativity_diffs():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    en_diffs = mcdlf.get_electronegativity_diffs()
    print(en_diffs)
    assert np.all(np.array(en_diffs) - np.array([1.21, 1.61, 1.33, 1.61]) < EPS)


def test_get_electronegativity_features():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    en_features = mcdlf.get_electronegativity_features()
    assert np.all(np.array(en_features) - np.array([5.76, 1.21, 1.61]) < EPS)


def test_get_coordinating_atom_numbers():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    print(mcdlf.get_coordinating_atom_numbers())
    assert mcdlf.get_coordinating_atom_numbers() == [7, 8, 17, 8]


def test_get_ligand_charges():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    assert mcdlf.get_ligand_charges() == [0, -1, -1, 0 , 0, -1, -2]


def test_get_ligand_denticity():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac"])
    assert mcdlf.get_ligand_denticity() == [4, 1, 1, 1, 1, 2]


def test_get_number_of_atoms():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac"])
    assert mcdlf.get_number_of_atoms() == 8


def test_get_ligand_number_of_atoms():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ligand_number_of_atoms = mcdlf.get_ligand_number_of_atoms()
    assert ligand_number_of_atoms == [2, 2, 1, 2]


def test_get_ligand_max_bond_order():
    if openbabel_available():
        mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
        assert int(np.max(mcdlf.get_ligand_max_bond_order(test_resource_dir + "furan.mol"))) == 2
        assert int(np.max(mcdlf.get_ligand_max_bond_order(test_resource_dir + "water.xyz"))) == 1
    else:
        pass


def test_get_kier_index():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ki = mcdlf.get_kier_index()
    assert ki - 3.11 < EPS


def test_get_kier_index_truncated():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ki = mcdlf.get_kier_index(1)
    assert ki - 1.0 < EPS


def test_get_all_ligands_atom_counts():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ligand_atom_counts = mcdlf.get_all_ligands_atom_counts()
    assert ligand_atom_counts == [0, 0, 2, 2, 0, 0, 0, 1, 0, 0]


def test_get_all_ligands_atom_counts_truncated():
    mcdlf = MCDL53(get_dummy_graph(), 3, ["12crown4", "chloride", "fluoride", "pph3", "phosphine", "acac", "s2-"])
    ligand_atom_counts = mcdlf.get_all_ligands_atom_counts(1)
    assert ligand_atom_counts == [0, 0, 1, 2, 0, 0, 0, 1, 0, 0]
