import re
import ast
import scipy
import numpy as np
import networkx as nx

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


class MCDL46():
    def __init__(self, graph: nx.graph, oxidation_state: int, ligand_list: list, multiplicity: int = None, truncation: int = 3) -> None:
        # sub_n denotes a non-scalar value
        self.graph = graph
        self.ligand_list = ligand_list
        self.feature_dict = {}
        self.metal_node_id = get_metal_node_id(self.graph)
        if self.metal_node_id == None: raise TypeError("Can not generate MCDL46 features without central metal")
        self.metal_identity = graph.nodes[self.metal_node_id]["atomic_number"]
        self.oxidation_state = oxidation_state
        self.electronegativity = electronegativity[self.metal_identity]
        # only for classifier where you do NOT use a pair of HS/LS TMCs (with two different multiplicities)
        if multiplicity:
            self.multiplicity = multiplicity
            # spin state one-hot encoding (LS: 0; HS: 1)
            self.spin_state = 0 if multiplicity < 2 else 1
        else:
            self.multiplicity = np.nan
            self.spin_state = np.nan
        # TODO: this is an 4 array
        self.connection_atom_n = self.get_coordinating_atom_numbers()
        self.ligand_charge_n = self.get_ligand_charges()
        self.ligand_denticity_n = self.get_ligand_denticity()
        self.ligands_as_subgraph_n = self.get_ligands_as_subgraph()
        self.total_number_of_atoms = self.get_number_of_atoms()
        self.ligand_number_of_atoms_n = self.get_ligand_number_of_atoms()
        # self.ligand_max_bond_order_n = self.get_ligand_max_bond_order()
        self.kier_index = self.get_kier_index()
        # self.truncated_kier_index = self.get_kier_index(truncation)
        # self.individual_atom_counts_n = self.get_all_ligands_atom_counts()
        # self.truncated_individual_atom_counts_n = self.get_all_ligands_atom_counts(truncation)



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


    def get_coordinating_atom_numbers(self) -> int:
        # this returns the atomic number of the metal coordinating atoms
        coord_atomic_numbers = []
        this_atoms_neighbors = self.graph.neighbors(self.metal_node_id)
        for bound_atoms in this_atoms_neighbors:
            coord_atomic_numbers += [self.graph.nodes[bound_atoms]["atomic_number"]]
        return coord_atomic_numbers


    def get_classifier_features(self, additional_featurizer: list = []) -> np.ndarray:
        """
        get features for a classifier task

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
        # TODO: for loop additional features
        pass


    def get_SSE_prediction_features(self, additional_featurizer: list=[]) -> np.ndarray:
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
        # TODO: for loop additional features
        pass


    def get_electronegativity_diffs(self) -> list:
        delta_ens = []
        this_atoms_neighbors = self.graph.neighbors(self.metal_node_id)
        for bound_atoms in this_atoms_neighbors:
            en_metal = electronegativity[self.graph.nodes[self.metal_node_id]["atomic_number"]]
            en_bound = electronegativity[self.graph.nodes[bound_atoms]["atomic_number"]]
            this_delEN =  en_bound - en_metal
            delta_ens += [this_delEN]
        return delta_ens



    def get_kier_index(self, truncation=None) -> float:
        if truncation is not None:
            graph = nx.generators.ego.ego_graph(self.graph, self.metal_node_id, truncation)
        else:
            graph = self.graph
        A = scipy.sparse.lil_matrix(nx.linalg.graphmatrix.adjacency_matrix(graph))
        n = A.shape[0]
        A *= A
        A.setdiag(0)
        p2 = A.sum() / 2
        n2 = n*n
        n3 = n2*n 
        if p2 != 0:
            return ((n3 - 5 * n2 + 8 * n - 4) / (p2*p2)) 
        else:
            return 0.0

    def get_ligands_as_subgraph(self) -> list:
        if self.metal_node_id == None:
            raise Exception("Could not find metal in complex.")
        connecting_atoms = list(self.graph.neighbors(self.metal_node_id))
        # Then cut the graph by removing all connections to the first atom
        subgraphs = self.graph.copy()
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
        if truncation is not None:
            graph = nx.generators.ego.ego_graph(self.graph, self.metal_node_id, truncation)
        else:
            graph = self.graph
        ligands = self.get_ligands_as_subgraph(graph)
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
        return counts_of_elements[mask]


    def get_number_of_atoms(self) -> int:
        return self.graph.number_of_nodes() 
        

    def get_ligand_number_of_atoms(self) -> list:
        ligand_sizes_n = [ligand[1].number_of_nodes() for ligand in self.ligands_as_subgraph_n]
        return ligand_sizes_n
        


    def get_mcdl46_features(self, graph, name, BO_graph=None):
        """returns mcdl46 (mcdl25) (10.1039/C7SC01247K) features for given TMC graph
            TODO: we leave out the bond order for now to avoid dependency on openbabe for now
        """
        # extract string based descriptors
        split = name.split("_") 
        ox_state = int(split[3]) 
        dents = []
        lig_charges = []
        lig_idents = []
        for lig in split[7:11]:
            lig_idents += [lig]
            dent_info = ligand_dict[lig][2]
            dents += [1 if (type(dent_info) == str) else len(dent_info)]
            lig_charges += [int(ligand_dict[lig][5][0])]
        spin = int(split[5]) 
        
        print(split, ox_state, dents, lig_charges, spin)

        metal_id = self.get_metal_id(graph)
        metal_identity = graph.nodes[metal_id]["atomic_number"]
    
        delta_ENs = self.get_electronegativity_diffs(graph, metal_id)
        sum_delEN = np.sum(delta_ENs)
        min_delEN = np.amin(delta_ENs)
        max_delEN = np.amax(delta_ENs)


        coord_atomic_numbers = self.get_coordinating_atom_numbers(graph, metal_id)

        kier_index = self.get_kier_index(graph)
        trunc_kier = self.get_kier_index(graph, 3, metal_id)
        
        num_atoms = graph.number_of_nodes() 
        
        ligands = self.get_ligands_as_subgraph(graph) 
        ligand_sizes = [ligand[1].number_of_nodes() for ligand in ligands] 
        ligand_bincount = self.get_ligand_atom_type_bincounts(graph)
        ligand_bincount_trunc = self.get_ligand_atom_type_bincounts(graph, 3, metal_id)

    
        spin_state = 0 if spin < 2 else 1

        if BO_graph != None:
            # this tedious procedure is necessary because we can not guarantee that nx_graph and molSimplify have the same properties
            metal = BO_graph.findMetal(transition_metals_only=True)
            A = BO_graph.getBondedAtoms(metal[0])
            # print(A)
            coord_atoms =  np.array(A)#.nonzero()[0]
            BO_graph.convert2OBMol()
            BOMatrix = BO_graph.populateBOMatrix()
            print(BOMatrix)
            # TODO: fix for some molecules
            max_bond_order = np.amax(BOMatrix[A])

            #feature_names = np.array(["metal_ident", "ox_state", "sum_dipole", "min_dipole", "max_dipole", "spin", "spin_state", "coord_atom", "lig_charge", "lig_denticity", "lig_#atoms", "lig_BO", "kier", "trunc_kier", "mul_metal", "mul_coord_atom"])
            feature_names = np.array(["I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "max(LBO)", "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
            mcdl46 =  np.array([metal_identity, ox_state, sum_delEN, min_delEN, max_delEN, spin, spin_state, *coord_atomic_numbers, *lig_charges, *dents, *ligand_sizes, max_bond_order, kier_index, trunc_kier, *ligand_bincount, *ligand_bincount_trunc])
            return mcdl46

        feature_names = np.array(["I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
        mcdl46 = np.array([metal_identity, ox_state, sum_delEN, min_delEN, max_delEN, spin, spin_state, *coord_atomic_numbers, *lig_charges, *dents, *ligand_sizes, kier_index, trunc_kier, *ligand_bincount, *ligand_bincount_trunc])
        return mcdl46


def get_target_property_from_row(row, target_props):
    if type(target_props) == list or isinstance(target_props, np.ndarray):
        props = []
        print(target_props)
        for target_prop in target_props:
            print(row[target_prop])
            props += [row[target_prop]]
        print(props)
        return props
    else:
        if target_props == "minBL.ls":
            return np.min(ast.literal_eval(re.sub(' +', ',', row["abs_bl.ls"])))
        if target_props == "minBl.hs":
            return np.min(ast.literal_eval(re.sub(' +', ',', row["abs_bl.hs"])))
        if target_props == "maxBL.ls":
            return np.max(ast.literal_eval(re.sub(' +', ',', row["abs_bl.ls"])))
        if target_props == "maxBl.hs":
            return np.max(ast.literal_eval(re.sub(' +', ',', row["abs_bl.hs"])))
        if target_props == "avgBL.ls":
            return np.mean(ast.literal_eval(re.sub(' +', ',', row["abs_bl.ls"])))
        if target_props == "avgBl.hs":
            return np.mean(ast.literal_eval(re.sub(' +', ',', row["abs_bl.hs"])))
        if target_props == "4BL.ls":
            return ast.literal_eval(re.sub(' +', ',', row["abs_bl.ls"]))
        if target_props == "4BL.hs":
            return ast.literal_eval(re.sub(' +', ',', row["abs_bl.hs"]))
        if target_props == "minrelBL.ls":
            return np.min(ast.literal_eval(re.sub(' +', ',', row["rel_bl.ls"])))
        if target_props == "minrelBl.hs":
            return np.min(ast.literal_eval(re.sub(' +', ',', row["rel_bl.hs"])))
        if target_props == "maxrelBL.ls":
            return np.max(ast.literal_eval(re.sub(' +', ',', row["rel_bl.ls"])))
        if target_props == "maxrelBl.hs":
            return np.max(ast.literal_eval(re.sub(' +', ',', row["rel_bl.hs"])))
        if target_props == "avgrelBL.ls":
            return np.mean(ast.literal_eval(re.sub(' +', ',', row["rel_bl.ls"])))
        if target_props == "avgrelBl.hs":
            return np.mean(ast.literal_eval(re.sub(' +', ',', row["rel_bl.hs"])))
        if target_props == "4relBL.ls":
            return ast.literal_eval(re.sub(' +', ',', row["rel_bl.ls"]))
        if target_props == "4relBL.hs":
            return ast.literal_eval(re.sub(' +', ',', row["rel_bl.hs"]))
        return row[target_props]