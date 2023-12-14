import numpy as np
from networkx import graph
from fcTMCml.featurizer.mol_graph_tools import graph_from_xyz_file
from fcTMCml.constants import chemical_symbols, electronegativity, test_resource_dir
from fcTMCml.featurizer.RAC import RAC


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


def test_tetrahedral_racs_vs_molSimplify():

    rac_array = np.load(
        test_resource_dir + "tetrahedral_racs/rac_features.npy", allow_pickle=True
    )

    # Since molSimplify uses different values for the covalent radii a
    # custom property function has to be used:
    covalent_radii = {
        "H": 0.37,
        "C": 0.77,
        "N": 0.75,
        "O": 0.73,
        "S": 1.02,
        "F": 0.71,
        "Cr": 1.27,
        "Mn": 1.39,
        "Fe": 1.25,
        "Co": 1.26,
    }

    # see above
    def property_fun(graph, node):
        output = np.zeros(5)
        Z = graph.nodes[node]["atomic_number"]
        # property (i): nuclear charge Z
        output[0] = Z
        # property (ii): Pauling electronegativity chi
        output[1] = electronegativity[Z]
        # property (iii): topology T, coordination number
        output[2] = len(list(graph.neighbors(node)))
        # property (iv): identity
        output[3] = 1.0
        # property (v): covalent radius S
        output[4] = covalent_radii[chemical_symbols[Z]]
        return output

    for name, _, racs_ref in rac_array:
        graph = graph_from_xyz_file(test_resource_dir + f"/tetrahedral_racs/xyz_files/{name}")
        rac150 = RAC(graph=graph, property_fun=property_fun, oxidation_state=2, ligand_list=["water", "chloride", "iodide", "carbonyl"])
        racs = rac150.get_tetrahedral_racs(depth=4, averaged=True)
        assert len(racs.flatten()) == 150
        print(racs.flatten()[:20])
        print(racs_ref[5:25])
        np.testing.assert_allclose(sorted(racs.flatten()), sorted(racs_ref[5:]))


def test_get_tetrahedral_feature_names():
    rac = RAC(graph=get_dummy_graph(), oxidation_state=2, ligand_list=["water", "chloride", "iodide", "carbonyl"])
    feature_names = rac.get_tetrahedral_rac_names()
    assert len(feature_names) == 300
    feature_names = rac.get_tetrahedral_rac_names(depth=4, averaged=True)
    assert len(feature_names) == 150
