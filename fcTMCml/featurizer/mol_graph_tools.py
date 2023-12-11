import numpy as np
import networkx as nx
from ase.io import read
from fcTMCml.constants import covalent_radii, metal_list


def graph_from_ase_atoms(atoms, threshold=1.2, covalent_radii=covalent_radii):
    g = nx.Graph()
    for i, atom in enumerate(atoms):
        g.add_node(i, symbol=atom.symbol)

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
