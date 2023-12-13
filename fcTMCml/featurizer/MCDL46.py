import scipy
import numpy as np
import networkx as nx
import importlib.util

from fcTMCml.constants import electronegativity
from fcTMCml.featurizer.mol_graph_tools import get_metal_node_id

"""
This class builds MCDL25 feauteres as described in DOI: 10.1039/c7sc01247k.
    - the exchange sensitivity is removed.
    - the ligand identity is removed
We add multiplicity, spin-state, the full Kier index, and (truncated) atom
counts to arrive at a MCDL46 feature vector.

---------------------------------------------------------
Scope   | Feature                   | Abbreviation      |
---------------------------------------------------------
Metal   | Identitiy                 | I(M)              |
        | Oxidation State           | Ox                |
        | Pauling Electronegativity | min/max/sum(chi)  |
        | Multiplicity              | S                 |*new(classifier only)
        | Spin State (HS vs. LS)    | SS                |*new(classifier only)
---------------------------------------------------------
Ligand  | Connection Atom           | CA                |
        | Charge                    | LC                |
        | Denticity                 | LD                |
        | Number of Atoms           | #A                |
        | Ligand Number of Atoms    | L#A               |*new
        | Bond Order                | max(LBO)          |(openbabel only)
        | Kier Index                | K                 |*new
        | Truncated Kier Index      | TK                |
---------------------------------------------------------
Counts  | Individual Atom Counts    | #                 |*new
        | Truncated Atom Counts     | T#                |*new
---------------------------------------------------------
"""


# load molSimplify ligand dict from ligands.dict
# TODO(ralf): is there a simpler way of doing this w/o using molSimplify?
ligand_dict = {x.split(":")[0]: x.split(":")[1][:-1].split(",") for x in open("fcTMCml/featurizer/ligands.dict").readlines()[2:]}


def openbabel_available() -> bool:
    # checks if openbabel is installed
    openbabel_available = importlib.util.find_spec("openbabel")
    return openbabel_available is not None


class MCDL46():
    def __init__(self, graph: nx.graph, oxidation_state: int, ligand_list: list, multiplicity: int = None, truncation: int = 3, input_file: str = None) -> None:
        """
        initialize MCDL46 features

        Parameters
        ----------
        input_file: str
            specify xyz file to construct OBMol for BOMatrix
        """
        # sub_n denotes a non-scalar value
        # following steps are preprocessing for actual feature extraction
        # (not included in the final features)
        self.graph = graph
        self.ligand_list = ligand_list
        self.metal_node_id = get_metal_node_id(self.graph)
        if self.metal_node_id is None:
            raise TypeError("Can not generate MCDL46 features without central metal")
        # start of feature extraction
        # metal (related) features
        self.metal_identity = graph.nodes[self.metal_node_id]["atomic_number"]
        self.oxidation_state = oxidation_state
        self.electronegativity_features_n = self.get_electronegativity_features()
        # only for classifier where you do NOT use a pair of HS/LS TMCs (with two different multiplicities)
        self.multiplicity = np.nan
        self.spin_state = np.nan
        if multiplicity:
            self.multiplicity = multiplicity
            # spin state one-hot encoding (LS: 0; HS: 1)
            self.spin_state = 0 if multiplicity < 2 else 1
        # ligand (related) features
        self.connection_atom_n = self.get_coordinating_atom_numbers()
        self.ligand_charge_n = self.get_ligand_charges()
        self.ligand_denticity_n = self.get_ligand_denticity()
        self.total_number_of_atoms = self.get_number_of_atoms()
        self.ligand_number_of_atoms_n = self.get_ligand_number_of_atoms()
        if openbabel_available() and input_file is not None:
            self.ligand_max_bond_order = self.get_ligand_max_bond_order(input_file)
        self.kier_index = self.get_kier_index()
        self.truncated_kier_index = self.get_kier_index(truncation)
        # count features
        self.individual_atom_counts_n = self.get_all_ligands_atom_counts()
        self.truncated_individual_atom_counts_n = self.get_all_ligands_atom_counts(truncation)

    ##############################
    # Feature Assembly Functions #
    ##############################

    def get_classifier_feature_names(self, additional_featurizer: list = []) -> list:
        feature_names = ["I(M)",
                         "Ox",
                         r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)",
                         "S",
                         "SS",
                         *["CA"] * len(self.connection_atom_n),
                         *["LC"] * len(self.ligand_charge_n),
                         *["LD"] * len(self.ligand_denticity_n),
                         "#A",
                         *["L#A"] * len(self.ligand_number_of_atoms_n),
                         ]
        if openbabel_available():
            feature_names += ["max_LBO"]
        feature_names += ["K",
                          "TK",
                          "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I",
                          "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"
                          ]
        for featurizer in additional_featurizer:
            feature_names += featurizer.get_classifier_feature_names()

        return feature_names

    def get_regression_feature_names(self, additional_featurizer: list = []) -> list:
        feature_names = ["I(M)",
                         "Ox",
                         r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)",
                         *["CA"] * len(self.connection_atom_n),
                         *["LC"] * len(self.ligand_charge_n),
                         *["LD"] * len(self.ligand_denticity_n),
                         "#A",
                         *["L#A"] * len(self.ligand_number_of_atoms_n)
                         ]
        if openbabel_available():
            feature_names += ["max_LBO"]
        feature_names += ["K",
                          "TK",
                          "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I",
                          "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"
                          ]
        for featurizer in additional_featurizer:
            feature_names += featurizer.get_classifier_feature_names()

        return feature_names

    def get_classifier_features(self, additional_featurizer: list = []) -> list:
        """
        get features for a classifier task
        returns mcdl46 (mcdl25) (10.1039/C7SC01247K) features for given TMC graph

        Parameters
        ----------
        additional_featurizer: list
            list of featurizers; features will be appended
            to feature vector in given order

        Returns
        -------
        feature_vector: np.array
            array with features for given transition metal complex
        """
        feature_array = [self.metal_identity,
                         self.oxidation_state,
                         *self.electronegativity_features_n,
                         # only for classifier where you do NOT use a pair of HS/LS TMCs (with two different multiplicities)
                         self.multiplicity,
                         self.spin_state,
                         self.connection_atom_n,
                         self.ligand_charge_n,
                         self.ligand_denticity_n,
                         self.total_number_of_atoms,
                         self.ligand_number_of_atoms_n
                         ]
        if openbabel_available():
            feature_array += self.ligand_max_bond_order
        feature_array += [self.kier_index,
                          self.truncated_kier_index,
                          *self.individual_atom_counts_n,
                          *self.truncated_individual_atom_counts_n
                          ]
        
        for featurizer in additional_featurizer:
            feature_array += featurizer.get_classifier_features()

        return feature_array

    def get_regression_features(self, additional_featurizer: list = []) -> np.ndarray:
        """
        get features for a prediction task

        Parameters
        ----------
        additional_featurizer: list
            list of featurizers; features will be appended
            to feature vector in given order

        Returns
        -------
        feature_vector: np.array
            array with features for given transition metal complex
        """
        feature_array = [self.metal_identity,
                         self.oxidation_state,
                         *self.electronegativity_features_n,
                         # only for classifier where you do NOT use a pair of HS/LS TMCs (with two different multiplicities)
                         self.connection_atom_n,
                         self.ligand_charge_n,
                         self.ligand_denticity_n,
                         self.total_number_of_atoms,
                         self.ligand_number_of_atoms_n
                         ]
        if openbabel_available():
            feature_array += self.ligand_max_bond_order
        feature_array += [self.kier_index, 
                          self.truncated_kier_index,
                          *self.individual_atom_counts_n,
                          *self.truncated_individual_atom_counts_n
                          ]
        
        for featurizer in additional_featurizer:
            feature_array += featurizer.get_regression_features()

        return feature_array

    ####################
    # Helper Functions #
    ####################

    def get_truncated_graph(self, truncation: int = None):
        if truncation is not None:
            return nx.generators.ego.ego_graph(self.graph, self.metal_node_id, truncation)
        else:
            return self.graph

    def get_ligands_as_subgraph(self, truncation: int = None) -> list:
        graph = self.get_truncated_graph(truncation)
        if self.metal_node_id is None:
            raise Exception("Could not find metal in complex.")
        connecting_atoms = list(graph.neighbors(self.metal_node_id))
        # Then cut the graph by removing all connections to the first atom
        subgraphs = graph.copy()
        subgraphs.remove_edges_from([(0, c) for c in connecting_atoms])
        # Build lists of connecting atom and ligand
        # subgraph tuples by first finding set of nodes for the component that the
        # connecting atom c comes from (using nx.node_conncted_component()) and
        # then constructing a subgraph using this node set.
        ligands = [
            (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
            for c in connecting_atoms
        ]
        return ligands

    ################################
    # Feature Extraction Functions #
    ################################

    def get_electronegativity_diffs(self) -> list:
        delta_ens = []
        this_atoms_neighbors = self.graph.neighbors(self.metal_node_id)
        for bound_atoms in this_atoms_neighbors:
            en_metal = electronegativity[self.graph.nodes[self.metal_node_id]["atomic_number"]]
            en_bound = electronegativity[self.graph.nodes[bound_atoms]["atomic_number"]]
            this_delEN = en_bound - en_metal
            delta_ens += [this_delEN]
        return delta_ens

    def get_electronegativity_features(self) -> list:
        delta_ens = self.get_electronegativity_diffs()
        return [np.sum(delta_ens), np.min(delta_ens), np.max(delta_ens)]

    def get_coordinating_atom_numbers(self) -> int:
        # this returns the atomic number of the metal coordinating atoms
        coord_atomic_numbers = []
        this_atoms_neighbors = self.graph.neighbors(self.metal_node_id)
        for bound_atoms in this_atoms_neighbors:
            coord_atomic_numbers += [self.graph.nodes[bound_atoms]["atomic_number"]]
        return coord_atomic_numbers

    def get_ligand_charges(self) -> list:
        charges_n = []
        for ligand in self.ligand_list:
            charges_n += [int(ligand_dict[ligand][5])]
        return charges_n

    def get_ligand_denticity(self) -> list:
        denticity_n = []
        for ligand in self.ligand_list:
            denticity_n += [int(len(ligand_dict[ligand][2].split(" ")))]
        return denticity_n

    def get_number_of_atoms(self) -> int:
        return self.graph.number_of_nodes()

    def get_ligand_number_of_atoms(self) -> list:
        self.ligands_as_subgraph_n = self.get_ligands_as_subgraph()
        ligand_sizes_n = [ligand[1].number_of_nodes() for ligand in self.ligands_as_subgraph_n]
        return ligand_sizes_n

    # modified from molSimplify (https://github.com/hjkgrp/molSimplify/blob/07dffb1fa4a061a6645c2e4030fd82ea9a0f81e6/molSimplify/Classes/mol3D.py#L2472)
    def get_ligand_max_bond_order(self, input_file: str) -> int:
        """
        Populate the bond order matrix using openbabel.

        Parameters
        ----------
        input_file: str
            path of input mol or xyz

        Returns
        -------
        max_bond_order : int
            maximal bond order in molecule (for TMCs this is the maximal bond order of all ligands)
        """

        from openbabel import openbabel as ob
        from openbabel import pybel as pb

        mol = next(pb.readfile(input_file.split(".")[-1], input_file))
        n = len(mol.atoms)
        molBOMat = np.zeros((n, n))
        for bond in ob.OBMolBondIter(mol.OBMol):
            these_inds = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            this_order = bond.GetBondOrder()
            molBOMat[these_inds[0] - 1, these_inds[1] - 1] = this_order
            molBOMat[these_inds[1] - 1, these_inds[0] - 1] = this_order
        return int(np.max(molBOMat))

    def get_kier_index(self, truncation: int = None) -> float:
        graph = self.get_truncated_graph(truncation)
        A = scipy.sparse.lil_matrix(nx.linalg.graphmatrix.adjacency_matrix(graph))
        n = A.shape[0]
        A *= A
        A.setdiag(0)
        p2 = A.sum() / 2
        n2 = n * n
        n3 = n2 * n
        if p2 != 0:
            return np.round(((n3 - 5 * n2 + 8 * n - 4) / (p2 * p2)), 2)
        else:
            return 0.0

    def get_all_ligands_atom_counts(self, truncation: int = None) -> list:
        """
        counts B, C, N, O, F, P, S, Cl, Br, I for all ligands

        Parameters
        ----------
        truncation: int
            truncate after n atoms away from metal
        Returns
        -------
        individual_atom_counts: list
            counts
        """
        ligands = self.get_ligands_as_subgraph(truncation)
        ligands_atom_list = []
        # store list of all atomic numbers of every ligand in ligands_atom_list
        for _, ligand in ligands:
            ligands_atom_list += [*list(nx.get_node_attributes(ligand, 'atomic_number').values())]
        # mask atoms of interest
        mask = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B, C, N, O, F, P, S, Cl, Br, I
        # generalize to any ligand and any type of atom
        counts_of_elements = np.zeros(127, dtype=int)
        # count occurence of every atom type in all ligands
        counts = np.bincount(ligands_atom_list)
        # assign counts to general full array
        counts_of_elements[:len(counts)] = counts
        # return counts of elements for those of interest
        return list(counts_of_elements[mask])
