import numpy as np
from fcTMCml.constants import EPS, test_resource_dir
from fcTMCml.featurizer.mol_graph_tools import graph_from_ase_atoms, graph_from_xyz_file, get_metal_node_id
from networkx import graph
from ase.io import read


def get_water_graph():
    g_ref = graph.Graph()
    g_ref.add_nodes_from(
        [(0, {"atomic_number": 8}), (1, {"atomic_number": 1}), (2, {"atomic_number": 1})]
    )
    g_ref.add_edges_from([(0, 1), (0, 2)])
    return g_ref

def test_graph_from_ase_atoms():
    g_ref = get_water_graph()
    atoms = read(test_resource_dir + "water.mol")
    g = graph_from_ase_atoms(atoms)
    assert g.nodes == g_ref.nodes
    assert g.edges == g_ref.edges


def test_graph_from_xyz_file():
    g_ref = get_water_graph()
    g = graph_from_xyz_file(test_resource_dir + "water.xyz")
    assert g.nodes == g_ref.nodes
    assert g.edges == g_ref.edges


def test_get_metal_node_id():
    g = graph_from_xyz_file(test_resource_dir + "dummy.xyz")
    assert get_metal_node_id(g) == 1