import re
import ast
import numpy as np
import networkx as nx
#from tmc_tools.graphs.mol_graph_tools import  get_metal_id

from ..constants import electronegativity






ligand_dict = np.loadtxt("../../data/ligands.dict", delimiter=',')

print(ligand_dict)
quit()

class MCDL46():
    def __init__(self) -> None:
        pass

    def get_electronegativity_diffs(self, graph, metal_id):
        delta_ens = []
        this_atoms_neighbors = graph.neighbors(metal_id)
        for bound_atoms in this_atoms_neighbors:
            this_delEN = electronegativity[graph.nodes[bound_atoms]["atomic_number"]] - electronegativity[graph.nodes[metal_id]["atomic_number"]]  
            delta_ens += [this_delEN]
        return delta_ens


    def get_coordinating_atom_numbers(graph, metal_id):
        coord_atomic_numbers = []
        this_atoms_neighbors = graph.neighbors(metal_id)
        for bound_atoms in this_atoms_neighbors:
            coord_atomic_numbers += [graph.nodes[bound_atoms]["atomic_number"]]
        return coord_atomic_numbers

    def get_kier_index(graph, truncation=None, metal_id=0):
        if truncation is not None:
            graph = nx.generators.ego.ego_graph(graph, metal_id, truncation)
        A = nx.linalg.graphmatrix.adjacency_matrix(graph)
        n = A.shape[0]
        A *= A
        A.setdiag(0)
        p2 = A.sum() / 2
        n2 = n*n
        n3 = n2*n 
        if p2 != 0:
            return ((n3 - 5 * n2 + 8 * n - 4) / (p2*p2)) 
        else:
            return 0

    def get_ligands_as_subgraph(graph):
        metal_id = get_metal_id(graph)
        if metal_id == None:
            raise Exception("Could not find metal in complex.")
        connecting_atoms = list(graph.neighbors(metal_id))
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

    def get_ligand_atom_type_bincounts(graph, truncation=None, metal_id=0):
        if truncation is not None:
            graph = nx.generators.ego.ego_graph(graph, metal_id, truncation)
        ligands = get_ligands_as_subgraph(graph)
        ligand_atom_list = []
        for i, ligand in ligands:
            ligand_atom_list += [*list(nx.get_node_attributes(ligand, 'atomic_number').values())]
        # mask atoms of interest
        mask = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        # generalize to any ligand and any type of atom
        counts_of_elements = np.zeros(127, dtype=int)
        counts = np.bincount(ligand_atom_list)
        counts_of_elements[:len(counts)] = counts
        return counts_of_elements[mask]

    def get_mcdl46_features(graph, name, BO_graph=None):
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

        metal_id = get_metal_id(graph)
        metal_identity = graph.nodes[metal_id]["atomic_number"]
    
        delta_ENs = get_electronegativity_diffs(graph, metal_id)
        sum_delEN = np.sum(delta_ENs)
        min_delEN = np.amin(delta_ENs)
        max_delEN = np.amax(delta_ENs)


        coord_atomic_numbers = get_coordinating_atom_numbers(graph, metal_id)

        kier_index = get_kier_index(graph)
        trunc_kier = get_kier_index(graph, 3, metal_id)
        
        num_atoms = graph.number_of_nodes() 
        
        ligands = get_ligands_as_subgraph(graph) 
        ligand_sizes = [ligand[1].number_of_nodes() for ligand in ligands] 
        ligand_bincount = get_ligand_atom_type_bincounts(graph)
        ligand_bincount_trunc = get_ligand_atom_type_bincounts(graph, 3, metal_id)
        #print(ligand_bincount)
        #print(ligand_bincount_trunc)
        #print(name)

    
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