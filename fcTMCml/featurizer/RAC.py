import numpy as np
import networkx as nx
import operator
from fcTMCml.constants import electronegativity, covalent_radii
from fcTMCml.featurizer.mol_graph_tools import get_metal_node_id


class RAC():
    def __init__(self, graph, property_fun=None) -> None:
        self.graph = graph
        if property_fun is None:
            self.property_fun = self.racs_property_vector
        else:
            self.property_fun = property_fun
        self.metal_node_id = get_metal_node_id(graph=graph)
        if self.metal_node_id is None:
            raise Exception("Could not find metal in complex.")

    def racs_property_vector(self, graph, node) -> np.array:
        output = np.zeros(5)
        Z = self.graph.nodes[node]["atomic_number"]
        # property (i): nuclear charge Z
        output[0] = Z
        # property (ii): Pauling electronegativity chi
        output[1] = electronegativity[Z]
        # property (iii): topology T, coordination number
        output[2] = len(list(graph.neighbors(node)))
        # property (iv): identity
        output[3] = 1.0
        # property (v): covalent radius S
        output[4] = covalent_radii[Z]
        return output

    def atom_centered_AC(self, graph, starting_node, depth: int = 3, operation=operator.mul) -> np.array:
        # Generate all paths from the starting node to all possible nodes
        lengths = nx.single_source_shortest_path_length(
            graph, source=starting_node, cutoff=depth
        )
        p_i = self.property_fun(graph, starting_node)
        output = np.zeros((depth + 1, len(p_i)))
        for node, d_ij in lengths.items():
            p_j = self.property_fun(graph, node)
            output[d_ij] += operation(p_i, p_j)
        return output

    def multi_centered_AC(self, graph, depth: int = 3, operation=operator.mul) -> np.array:
        n_props = len(self.property_fun(graph, list(graph.nodes.keys())[0]))
        output = np.zeros((depth + 1, n_props))
        # Generate all pairwise path lengths
        lengths = nx.all_pairs_shortest_path_length(graph, cutoff=depth)
        for node_i, lengths_i in lengths:
            p_i = self.property_fun(graph, node_i)
            for node_j, d_ij in lengths_i.items():
                p_j = self.property_fun(graph, node_j)
                output[d_ij] += operation(p_i, p_j)
        return output

    def get_tetrahedral_racs(self, depth: int = 3, averaging=False):
        """
        compute RACs for tetrahedral TMCs without averaging

        Parameters
        ----------
        graph : networkX.graph
            TMC graph of molecule we want to generate RAC for
        depth (int):
            maximum depth of RAC (default: 3, default for oct: 4)
        property_fun : array like, optional
            properties to compute RACs for (default: racs_property_vector)
        """
        # Following J. Phys. Chem. A 2017, 121, 8939
        # For tetrahedrals there are 4 start/scope
        # combinations for product ACs and 2 for difference ACs.
        n_props = len(self.property_fun(self.graph, list(self.graph.nodes.keys())[0]))
        output = np.zeros((4 + 2, depth + 1, n_props)) if averaging else np.zeros((3 + 3 * 4, depth + 1, n_props))

        # start = f, scope = all, product
        output[0] = self.multi_centered_AC(self.graph, depth=depth)
        # start = mc, scope = all, product
        output[1] = self.atom_centered_AC(self.graph, self.metal_node_id, depth=depth)

        # For the other scopes the graph has to be subdivided into individual
        # ligand graphs. Make these changes on a copy of the graph:
        subgraphs = self.graph.copy()
        # First find all connecting atoms (assumes the center is node 0):

        connecting_atoms = list(subgraphs.neighbors(self.metal_node_id))
        # Assert that we are removing 4 edges
        if len(connecting_atoms) != 4:
            raise ValueError(
                "First entry in the graph does not have 4 neighbors "
                "as expected for an octahedral complex."
            )
        # Then cut the graph by removing all connections to the first atom
        subgraphs.remove_edges_from([(0, c) for c in connecting_atoms])

        # Build lists of connecting atom and ligand
        # subgraph tuples by first finding set of nodes for the component that the
        # connecting atom c comes from (using nx.node_conncted_component()) and
        # then constructing a subgraph using this node set.
        # TODO(jonas): move function from MCDL53 features to tools.py
        ligands = [
            (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
            for c in connecting_atoms
        ]

        # Note that the ligand centered RACs are averaged over the involved
        # ligands.
        if averaging:
            # start = lc, scope = lig, product
            output[2] = np.mean([self.atom_centered_AC(g, c, depth=depth) for (c, g) in ligands], axis=0)
            # start = lig, scope = lig, product
            output[3] = np.mean([self.multi_centered_AC(g, depth=depth) for (_, g) in ligands], axis=0)

            # Finally calculate the difference ACs the same way:
            # start = mc, scope = all, difference
            output[4] = self.atom_centered_AC(self.graph, 0, depth=depth, operation=operator.sub)
            # start = lc, scope = lig, difference
            output[5] = np.mean([self.atom_centered_AC(g, c, depth=depth, operation=operator.sub) for (c, g) in ligands], axis=0)
        else:
            # start = lc, scope = lig, product
            output[3:3 + 4] = [self.atom_centered_AC(g, c, depth=depth) for (c, g) in ligands]
            # start = lig, scope = lig, product
            output[7:7 + 4] = [self.multi_centered_AC(g, depth=depth) for (_, g) in ligands]
            # Finally calculate the difference ACs the same way:
            # start = mc, scope = all, difference
            output[2] = self.atom_centered_AC(self.graph, self.metal_node_id, depth=depth, operation=operator.sub)
            # start = lc, scope = lig, difference
            output[11:11 + 4] = [self.atom_centered_AC(g, c, depth=depth, operation=operator.sub) for (c, g) in ligands]

        return output
