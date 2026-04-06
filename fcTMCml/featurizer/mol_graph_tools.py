import numpy as np
import networkx as nx
from ase.io import read
from scipy.sparse import diags
from fcTMCml.constants import covalent_radii, metal_list


def graph_from_ase_atoms(atoms, threshold=1.2, covalent_radii=covalent_radii):
    g = nx.Graph()
    for i, atom in enumerate(atoms):
        g.add_node(i, atomic_number=atom.number)

    for i, ai in enumerate(atoms):
        for j, aj in enumerate(atoms[i + 1 :]):
            r = np.linalg.norm(ai.position - aj.position)
            if r < threshold * (covalent_radii[ai.number] + covalent_radii[aj.number]):
                g.add_edge(i, j + i + 1)
    return g


def graph_from_xyz_file(file, **kwargs):
    atoms = read(file)
    return graph_from_ase_atoms(atoms, **kwargs)


def get_metal_node_id(graph):
    for node in graph.nodes():
        if graph.nodes[node]['atomic_number'] in metal_list:
            return node
    return None


def compute_graph_determinant(graph):
    # compute graph determinant
    # according to https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.0c01458
    weights = diags(list(nx.get_node_attributes(graph, "atomic_number").values()))
    A = nx.adjacency_matrix(graph)
    weighted_A = weights @ A @ weights
    return np.linalg.det(weighted_A.todense())
