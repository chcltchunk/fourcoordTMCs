import re
import ast
import numpy as np
import networkx as nx
from tmc_tools.graphs.constructors import graph_from_xyz_file
from tmc_tools.graphs.racs import racs_property_vector, tetrahedral_racs, get_set_of_lig_scaled_RACs 
from tmc_tools.graphs.mol_graph_tools import  get_metal_id, compute_graph_determinant, compute_graph_infinity_norm
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

from tmc_tools.constants import electronegativity

from .DBLOC_inspired_energy_gap import DBLOC_inspired_energy_gap


from molSimplify.Classes.globalvars import globalvars, amassdict
from molSimplify.Scripts.io import getlicores
from molSimplify.Classes.mol3D import mol3D

globs = globalvars()


molsimplify_ligand_dict = getlicores()


def get_electronegativity_diffs(graph, metal_id):
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
       return ((n3 - 5 * n2 + 8 * n - 4)
                     / (p2*p2)) 
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
        dent_info = molsimplify_ligand_dict[lig][2]
        dents += [1 if (type(dent_info) == str) else len(dent_info)]
        lig_charges += [int(molsimplify_ligand_dict[lig][5][0])]
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

def get_mulliken_charges(target_path):
    # decided not to use that anymore
    charge_path = glob.glob(re.split("/", target_path)[-1]+"/scr/charge_mull.xls")
    print(charge_path)
    charges = np.genfromtxt(charge_path).T[-1]
    charge_mask = np.array([0] + A)
    # print(charges)
    # print(charge_mask)
    charges = charges[charge_mask][:2]
    return charges

def extend_racs(
    rac_set,
    per_rack_identifiers,
    extension_type='default'
):
    """
    works for identifiers of the format:
        metal_co_ox_2_spin_2_ligstr_ligand1_ligand2_ligand3_ligand4_*
    adds 9 additional elements to feature vector
    extension_type : str
        type of extension: 'default', 'phys_ediffs', 'phys_ediffs_only' 'dbloc_ediffs'
    """
    extensions = []
    for index, name in enumerate(per_rack_identifiers):
        split = name.split("_") 
        metal = str(split[1])
        ox = int(split[3]) 
        mult = int(split[5]) 
        lig_list = split[7:11]
        dents = []
        lig_charges = []
        for lig in lig_list:
            dent_info = molsimplify_ligand_dict[lig][2]
            dents += [1 if (type(dent_info) == str) else len(dent_info)]
            lig_charges += [int(molsimplify_ligand_dict[lig][5][0])]
        if extension_type == 'default':
            extensions += [[ox, mult, *dents]]
        elif extension_type == 'phys_ediffs':
            dbloc = DBLOC_inspired_energy_gap(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ox, ediff, guess, *lig_charges, *dents]]
        elif extension_type == 'phys_ediffs_only':
            dbloc = DBLOC_inspired_energy_gap(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ediff, guess]]
        elif extension_type == 'dbloc_ediffs':
            dbloc = DBLOC_inspired_energy_gap(metal, ox, mult, lig_list)
            occ = dbloc.occupy_d_orbitals()
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=True)
            extensions += [[ox, ediff, guess, *lig_charges, *dents]]
        else:
            raise ValueError("define a proper extension type: default, phys_ediffs, 'phys_ediffs_only', dbloc_ediffs")
    extensions = np.vstack(extensions)
    rac_set = rac_set.reshape(-1, np.prod(rac_set.shape[1:]))
    extended_racs = np.hstack((extensions, rac_set))
    return extended_racs


def extend_mcdl46(
    mcdl46_set,
    per_mcdl46_identifiers,
    extension_type='phys_ediffs'
):
    """
    works for identifiers of the format:
        metal_co_ox_2_spin_2_ligstr_ligand1_ligand2_ligand3_ligand4_*
    adds 9 additional elements to feature vector
    extension_type : str
        type of extension: 'phys_ediffs', 'dbloc_ediffs'
    """
    extensions = []
    for index, name in enumerate(per_mcdl46_identifiers):
        split = name.split("_") 
        metal = str(split[1])
        ox = int(split[3]) 
        mult = int(split[5]) 
        lig_list = split[7:11]
        dents = []
        lig_charges = []
        for lig in lig_list:
            dent_info = molsimplify_ligand_dict[lig][2]
            dents += [1 if (type(dent_info) == str) else len(dent_info)]
            lig_charges += [int(molsimplify_ligand_dict[lig][5][0])]
        if extension_type == 'phys_ediffs':
            dbloc = DBLOC_inspired_energy_gap(metal, ox, mult, None)
            occ = dbloc.occupy_d_orbitals()
            #ediff = dbloc.get_energy_diff_sqp_thd(dbloc=False)
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=False)
            extensions += [[ediff, guess]]
        elif extension_type == 'dbloc_ediffs':
            dbloc = DBLOC_inspired_energy_gap(metal, ox, mult, lig_list)
            occ = dbloc.occupy_d_orbitals()
            ediff, guess = dbloc.get_energy_diff_sqp_thd_and_geom_guess(dbloc=True)
            extensions += [[ediff, guess]]
        else:
            raise ValueError("define a proper extension type: phys_ediffs, dbloc_ediffs")
    extensions = np.vstack(extensions)
    extended_racs = np.hstack((extensions, mcdl46_set.reshape(-1, np.prod(mcdl46_set.shape[1:]))))
    print(extended_racs.shape)
    return extended_racs

dummy = np.random.rand(5, 15, 20)
dummy_identifiers = [
    "metal_co_ox_2_spin_2_ligstr_acetonitrile_acetonitrile_acetonitrile_acetonitrile_charge_nan",
    "metal_co_ox_3_spin_2_ligstr_acetonitrile_acetonitrile_acetonitrile_acetonitrile_charge_nan",
    "metal_mn_ox_5_spin_2_ligstr_acetonitrile_acetonitrile_acetonitrile_acetonitrile_charge_nan",
    "metal_cr_ox_2_spin_2_ligstr_acetonitrile_acetonitrile_acetonitrile_acetonitrile_charge_nan",
    "metal_co_ox_1_spin_2_ligstr_acetonitrile_acetonitrile_acetonitrile_acetonitrile_charge_nan",
]
print(extend_racs(dummy, dummy_identifiers))

def apply_restrictions_on_dataset_sample(sample, restrictions, spin_state=None):
    sample_approved = True
    if "geometry" in restrictions.keys():
        #sample_approved = sample_approved and (sample["geom.ls"] in restrictions["geometry"]) and (sample["geom.hs"] in restrictions["geometry"])
        if spin_state == "ls":
            sample_approved = sample_approved and (sample["geom.ls"] in restrictions["geometry"])
        elif spin_state == "hs":
            sample_approved = sample_approved and (sample["geom.hs"] in restrictions["geometry"])
        else:
            sample_approved = sample_approved and (sample["geom.ls"] in restrictions["geometry"]) or (sample["geom.hs"] in restrictions["geometry"])
    if "tetrahedral" in restrictions.keys():
        sample_approved = sample_approved and (sample["tetrahedral.ls"] == restrictions["tetrahedral"]) and (sample["tetrahedral.hs"] == restrictions["tetrahedral"])
    if "homo_smaller_than" in restrictions.keys():
        sample_approved = sample_approved and (sample["homo.ls"] < restrictions["homo_smaller_than"]) and (sample["homo.hs"] < restrictions["homo_smaller_than"])
    if "s2_cutoff" in restrictions.keys():
        sample_approved = sample_approved and (np.abs(sample["s2_is.ls"]-sample["s2_expect.ls"]) < restrictions["s2_cutoff"]) and (np.abs(sample["s2_is.hs"]-sample["s2_expect.hs"]) < restrictions["s2_cutoff"])
    if "old_new_sse_diff_smaller_than" in restrictions.keys():
        sample_approved = sample_approved and (np.abs(sample["b3lyp.sse (kcal/mol)"]-sample["old_sse_if available"]) < restrictions["old_new_sse_diff_smaller_than"])
    if "assort_by_name" in restrictions.keys():
        #print("metal_"+sample["metal"]+"_ox_"+str(int(sample["ox"]))+"_spin_"+str(int(sample["ls.spin"]))+"_ligstr_"+sample["ligstr"]+"_charge_nan")
        sample_approved = sample_approved and ("metal_"+sample["metal"]+"_ox_"+str(int(sample["ox"]))+"_spin_"+str(int(sample["ls.spin"]))+"_ligstr_"+sample["ligstr"]+"_charge_nan" not in restrictions["assort_by_name"])
    if "restrict_coord_atoms_list" in restrictions.keys():
        coord_atoms = list(ast.literal_eval(sample["coordAtom.ls"]).keys())
        for coord_atom in coord_atoms:
            sample_approved = sample_approved and (coord_atom in restrictions["restrict_coord_atoms_list"])
    if "restrict_ligands" in restrictions.keys():
        ligs = sample["ligstr"].split("_")
        for lig in ligs:
            sample_approved = sample_approved and (lig in restrictions["restrict_ligands"])
    return sample_approved

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
        
#quit()
def main():
    xyz_directory_homo = "./geometries_homo/"
    xyz_directory_hetero = "./geometries_hetero/"
    thd_df_homo = pd.read_csv("homoleptic_thd_sses_bl_homo_with_validation_data_no_spin_correction.ssv", delimiter=';')
    thd_df_homo = pd.read_csv("homoleptic_thd_sses_bl_homo_with_validation_data.ssv", delimiter=';')
    thd_df_hetero = pd.read_csv("heteroleptic_thd_sses_bl_homo_with_validation_data_no_spin_correction.ssv", delimiter=';')
    thd_df_hetero = pd.read_csv("heteroleptic_thd_sses_bl_homo_with_validation_data.ssv", delimiter=';')
    thd_df_homo = pd.read_csv("homoleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", delimiter=';')
    thd_df_hetero = pd.read_csv("heteroleptic_thd_sses_bl_homo_with_validation_data_exchange_sensitivity.ssv", delimiter=';')
    #thd_df_homo = pd.read_csv("homoleptic_thd_sses_with_validation_data.csv")
    #thd_df_hetero = pd.read_csv("heteroleptic_thd_sses_with_validation_data.csv")
    #print('found ' + str(len(target_paths_ls)) + ' ' + str(len(target_paths_hs)) + ' molecules to read')
    restrictions = {#"tetrahedral" : True,
                    #"restrict_coord_atoms_list" : ["O", "N", "F"],
                    "geometry" : ["square planar", "tetrahedral"],
                    #"restrict_ligands" : ['ox', 'chloride', 'scn', 'cn', 'cabonyl', 'ncs', 'ammonia', 'phen', 'bipy', 'en', 'water', 'pisc', 'misc', 'acac', 'tbuc'], 
                    #"restrict_ligands" : ['chloride', 'carbonyl', 'cn', 'acetonitrile', 'phen', 'bipy', 'en', 'ammonia', 'pyridine', 'methylamine', 'scn', 'water', 'acac', 'ox', 'hydroxyl', 'cyanate', 'fluoride', 'ncs', 'benzisc', 'cyanopyridine', 'formaldehyde'],
                    "restrict_ligands" : ['chloride', "flouride", 'carbonyl', 'cn', 'acetonitrile', 'phen', 'bipy', 'en', 'ammonia', 'pyridine', 'methylamine', 'scn', 'water', 'acac', 'ox', 'hydroxyl', 'cyanate', 'fluoride', 'ncs', 'benzisc', 'cyanopyridine', 'formaldehyde'], # mcdl46 only
                    #"homo_smaller_than" : 0,
                    #"s2_cutoff" : 1,
                    "old_new_sse_diff_smaller_than" : 30, 
                    #"assort_by_name" : ['metal_cr_ox_3_spin_2_ligstr_acac_acac_acac_acac_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_acac_acac_acac_acac_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_bipy_bipy_bipy_bipy_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_ox_ox_ox_ox_charge_nan', 'metal_mn_ox_3_spin_1_ligstr_phen_phen_phen_phen_charge_nan'], 
                   #"assort_by_name" : ['metal_cr_ox_3_spin_2_ligstr_acac_acac_acac_acac_charge_nan', 'metal_cr_ox_3_spin_2_ligstr_mec_mec_mec_mec_charge_nan', 'metal_cr_ox_3_spin_2_ligstr_ox_ox_ox_ox_charge_nan', 'metal_cr_ox_3_spin_2_ligstr_tbuc_tbuc_tbuc_tbuc_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_acac_acac_acac_acac_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_bipy_bipy_bipy_bipy_charge_nan', 'metal_fe_ox_3_spin_2_ligstr_ox_ox_ox_ox_charge_nan', 'metal_mn_ox_2_spin_2_ligstr_tbuc_tbuc_tbuc_tbuc_charge_nan', 'metal_mn_ox_3_spin_1_ligstr_aceticacidbipyridine_aceticacidbipyridine_aceticacidbipyridine_aceticacidbipyridine_charge_nan', 'metal_mn_ox_3_spin_1_ligstr_bipy_bipy_bipy_bipy_charge_nan', 'metal_mn_ox_3_spin_1_ligstr_mebipy_mebipy_mebipy_mebipy_charge_nan', 'metal_mn_ox_3_spin_1_ligstr_phen_phen_phen_phen_charge_nan']
                    #"assort_by_name" : ['metal_cr_ox_3_spin_2_ligstr_fluoride_hydroxyl_hydroxyl_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_fluoride_fluoride_fluoride_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_formaldehyde_formaldehyde_hydroxyl_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_formaldehyde_formaldehyde_formaldehyde_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_ncs_ncs_ncs_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_pyridine_hydroxyl_hydroxyl_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_cyanopyridine_hydroxyl_hydroxyl_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_cyanopyridine_cyanopyridine_cyanopyridine_hydroxyl_charge_nan','metal_cr_ox_3_spin_2_ligstr_thiopyridine_thiopyridine_thiopyridine_hydroxyl_charge_nan','metal_mn_ox_2_spin_2_ligstr_furan_furan_furan_hydroxyl_charge_nan','metal_mn_ox_2_spin_2_ligstr_thiopyridine_hydroxyl_hydroxyl_hydroxyl_charge_nan']

                }
    restrictions1 = {
                    "geometry" : ["tetrahedral"],
                    "restrict_ligands" : ['chloride', "flouride", 'carbonyl', 'cn', 'acetonitrile', 'phen', 'bipy', 'en', 'ammonia', 'pyridine', 'methylamine', 'scn', 'water', 'acac', 'ox', 'hydroxyl', 'cyanate', 'fluoride', 'ncs', 'benzisc', 'cyanopyridine', 'formaldehyde'], # mcdl46 only
                    "old_new_sse_diff_smaller_than" : 30, 
                    }
    restrictions2 = {"tetrahedral" : True,
                    "old_new_sse_diff_smaller_than" : 30, 
                    }
    print(restrictions)
    hs_ls_accumulated_property = True
    hs_ls_accumulated_property = False  # if false we genearate feature vector for high and low spin configuration each
    avail_target_props = np.array(["b3lyp.sse (kcal/mol)", "homo.ls", "homo.hs", "minBL.ls", "minBl.hs", "maxBL.ls", "maxBl.hs", "avgBL.ls", "avgBl.hs", "geom.ls", "geom.hs", "4BL.ls", "4BL.hs", "minrelBL.ls", "minrelBl.hs", "maxrelBL.ls", "maxrelBl.hs", "avgrelBL.ls", "avgrelBl.hs", "4relBL.ls", "4relBL.hs"], dtype=object)
    #target_props =  avail_target_props[0]
    target_props_list = [avail_target_props[0], avail_target_props[1], avail_target_props[3], avail_target_props[7], avail_target_props[11], avail_target_props[13], avail_target_props[15], avail_target_props[17], avail_target_props[19]] 
    target_props_hs_list = [None, avail_target_props[2], avail_target_props[4], avail_target_props[8], avail_target_props[12], avail_target_props[14], avail_target_props[16], avail_target_props[18], avail_target_props[20]]
    hs_ls_accumulated_property_list = [True, False, False, False, False, False, False, False, False]
    #target_props =  avail_target_props[[1, -2]]
    #target_props_hs =  avail_target_props[[2, -1]]
    restrictions_list = [restrictions, restrictions1, restrictions1, restrictions1, restrictions1, restrictions1, restrictions1, restrictions1, restrictions1]
    base_paths = ["feature_sets_SSE", "feature_sets_HOMO", "feature_sets_minBL", "feature_sets_avgBL",  "feature_sets_4BL",  "feature_sets_min_relBL", "feature_sets_avg_relBL", "feature_sets_4relBL"]
    base_paths = ["feature_sets_SSE_every_ligand_for_active_learning"]
    restrictions_list = [restrictions2]
    target_props_list = [avail_target_props[0]]
    target_props_hs_list = [None]
    hs_ls_accumulated_property_list = [True]
    for base_path, restrictions, target_props, target_props_hs, hs_ls_accumulated_property in zip(base_paths, restrictions_list, target_props_list, target_props_hs_list, hs_ls_accumulated_property_list):
        print(target_props)
        if type(target_props) == str:
            restrictions_descriptor = "_sse" if target_props == "b3lyp.sse (kcal/mol)" else "_"+target_props[:-3]
        else:
            print(target_props)
            restrictions_descriptor =  "_".join(["sse" if (target_prop == "b3lyp.sse (kcal/mol)") else target_prop[:-3] for target_prop in target_props])
        for i in restrictions.keys():
            if i == "assort_by_name": 
                continue
            if i == "restrict_ligands": 
                restrictions_descriptor += "_"+i+"_"+"dbloc_compatible_selection"
                continue
            #restrictions_descriptor += "_"+i+"_"+("_".join(restrictions[i]) if type(restrictions[i]) == list else str(restrictions[i]))
            restrictions_descriptor += "_"+i+"_"+("_".join(restrictions[i]) if type(restrictions[i]) == list else str(restrictions[i]))
        
        print(restrictions_descriptor)
        target_paths = []
        for q, i in thd_df_homo.iterrows():
            name = "metal_{}_ox_{}_spin_{}_ligstr_{}_charge_{}".format(i["metal"], str(int(i["ox"])), str(int(i["ls.spin"])), i["ligstr"], i["charge"])
            name_hs = "metal_{}_ox_{}_spin_{}_ligstr_{}_charge_{}".format(i["metal"], str(int(i["ox"])), str(int(i["hs.spin"])), i["ligstr"], i["charge"])
            # if (i["tetrahedral.ls"] == 1) and (i["tetrahedral.hs"] == 1) and (np.abs(i["s2_is.ls"]-i["s2_expect.ls"]) < 1.0) and (np.abs(i["s2_is.hs"]-i["s2_expect.hs"]) < 1.0):
            if apply_restrictions_on_dataset_sample(i, restrictions):
                target_paths += [[xyz_directory_homo+name+".xyz", get_target_property_from_row(i, target_props), i["metal"], i["ox"], set(i["ligstr"].split("_"))]]
                if not hs_ls_accumulated_property:
                    target_paths += [[xyz_directory_homo+name_hs+".xyz", get_target_property_from_row(i, target_props_hs), i["metal"], i["ox"], set(i["ligstr"].split("_"))]]
            

        for q, i in thd_df_hetero.iterrows():
            name = "metal_{}_ox_{}_spin_{}_ligstr_{}_charge_{}".format(i["metal"], str(int(i["ox"])), str(int(i["ls.spin"])), i["ligstr"], i["charge"])
            name_hs = "metal_{}_ox_{}_spin_{}_ligstr_{}_charge_{}".format(i["metal"], str(int(i["ox"])), str(int(i["hs.spin"])), i["ligstr"], i["charge"])
            #if (i["tetrahedral.ls"] == 1) and (i["tetrahedral.hs"] == 1) and (np.abs(i["s2_is.ls"]-i["s2_expect.ls"]) < 1.0) and (np.abs(i["s2_is.hs"]-i["s2_expect.hs"]) < 1.0):
            if apply_restrictions_on_dataset_sample(i, restrictions):
                target_paths += [[xyz_directory_hetero+name+".xyz", get_target_property_from_row(i, target_props), i["metal"], i["ox"], set(i["ligstr"].split("_"))]]
                if not hs_ls_accumulated_property:
                    target_paths += [[xyz_directory_hetero+name_hs+".xyz", get_target_property_from_row(i, target_props_hs), i["metal"], i["ox"], set(i["ligstr"].split("_"))]]

        print(len(target_paths))
        rac_features = []
        rac_mean_features = []
        rac_lig_scaled_features = []
        mcdl46_features = []
        mcdl46_features_complete = []
        rac_features_non_flat = []
        lig_trackers = []
        
        features = []
        for (xyz, target_prop, metal, ox, ligstr) in target_paths:
            print(xyz)
            if xyz in ["./geometries_hetero/metal_fe_ox_2_spin_1_ligstr_fluoride_iodide_iodide_iodide_charge_nan.xyz"]:
                continue
            #if target_prop[1] not in restrictions["geometry"]:
            #    continue
            print(target_prop)
            name = xyz.split('/')[2].split('.')[0]
            print(name)
            for thresh in np.linspace(1.1, 1.8, 11):
                graph = graph_from_xyz_file(xyz, threshold=thresh)
                if len(list(graph.neighbors(0))) == 4:
                    break
            molSimplify_graph = mol3D() # mol3D instance for BO feature
            molSimplify_graph.readfromxyz(xyz, read_final_optim_step=True) # read geo
            try:
                print(xyz)
                racs, lig_tracker = tetrahedral_racs(graph, depth=3, scaler='ligand_based')
                mean_racs = tetrahedral_racs(graph, depth=3, ligand_accumulation_operation=(np.mean, {"axis": 0}))
                lig_scaled_racs = get_set_of_lig_scaled_RACs(racs, lig_tracker) 
                #racs, lig_tracker = tetrahedral_racs(graph, depth=3, scaler='none')
                lig_trackers += [lig_tracker]
                mcdl46 = get_mcdl46_features(graph, name)
                mcdl46_complete = get_mcdl46_features(graph, name, BO_graph=molSimplify_graph)
                #print(mcdl46, name)
                features.append(racs.flatten())
                rac_lig_scaled_features += [[name, target_prop, lig_scaled_racs.flatten()]]
                rac_mean_features += [[name, target_prop, mean_racs.flatten()]]
                rac_features += [[name, target_prop, racs.flatten()]]
                mcdl46_features += [[name, target_prop, mcdl46.flatten()]]
                mcdl46_features_complete += [[name, target_prop, mcdl46_complete.flatten()]]
                rac_features_non_flat += [[name, target_prop, racs]]
            except ValueError:
                print('Wrong connectivity for:', xyz, list(graph.neighbors(0)))
        
        features = np.array(features)
        print(features.shape)
        
        hs_ls_accumulated_property = "true" if hs_ls_accumulated_property else "false"
        rac_features = np.array(rac_features, dtype=object)
        rac_mean_features = np.array(rac_mean_features, dtype=object)
        rac_lig_scaled_features = np.array(rac_lig_scaled_features, dtype=object)
        rac_features_non_flat = np.array(rac_features_non_flat, dtype=object)
        mcdl46_features = np.array(mcdl46_features, dtype=object)
        mcdl46_features_complete = np.array(mcdl46_features_complete, dtype=object)
        print(rac_features.shape)
        features = np.vstack(mcdl46_features_complete[:, 2])
        mcdl46_features_extended = extend_mcdl46(features, mcdl46_features_complete[:,0])
        #feature_names = np.array(["DelP" "GuessP", "I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "max(LBO)", "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
        np.save(base_path+"/"+"mcdl46_extended_phys_classifier"+restrictions_descriptor, np.concatenate((mcdl46_features_complete[:,0][:, np.newaxis],mcdl46_features_complete[:,1][:, np.newaxis], mcdl46_features_extended), axis=1))
        scaler = StandardScaler()
        scaler.fit(mcdl46_features_extended)
        scaled_features_mcdl46_extended = scaler.transform(mcdl46_features_extended)
        #feature_names = np.array(["DelP", "GuessP", "I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "max(LBO)", "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
        np.save(base_path+"/"+"mcdl46_extended_phys_scaled_classifier"+restrictions_descriptor, np.concatenate((mcdl46_features_complete[:,0][:, np.newaxis],mcdl46_features_complete[:,1][:, np.newaxis], scaled_features_mcdl46_extended), axis=1))
        #mcdl46_features_extended = extend_mcdl46(features, mcdl46_features_complete[:,0], extension_type='dbloc_ediffs')
        #feature_names = np.array(["DelDB", "GuessDB", "I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "max(LBO)", "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
        #np.save(base_path+"/"+"mcdl46_extended_dbloc_classifier"+restrictions_descriptor, np.concatenate((mcdl46_features_complete[:,0][:, np.newaxis],mcdl46_features_complete[:,1][:, np.newaxis], mcdl46_features_extended), axis=1))
        #scaler = StandardScaler()
        #scaler.fit(mcdl46_features_extended)
        #scaled_features_mcdl46_extended_dbloc = scaler.transform(mcdl46_features_extended)
        #feature_names = np.array(["DelDB", "GuessDB", "I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", *["CA"]*len(coord_atomic_numbers), *["LC"]*len(lig_charges), *["LD"]*len(dents), *["L#A"]*len(ligand_sizes), "max(LBO)", "K", "TK", "#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I", "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"])
        #np.save(base_path+"/"+"mcdl46_extended_dbloc_scaled_classifier"+restrictions_descriptor, np.concatenate((mcdl46_features_complete[:,0][:, np.newaxis],mcdl46_features_complete[:,1][:, np.newaxis], scaled_features_mcdl46_extended_dbloc), axis=1))
        scaled_features = features
        #scaler = StandardScaler()
        #scaler.fit(features)
        #scaled_features = scaler.transform(features)
        feature_names = "rac_features_depth_3" 
         
        scaled_features = np.vstack(rac_lig_scaled_features[:, 2])
        np.save(base_path+"/"+"rac_features_new_lig_scaled"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features), axis=1))
        extended_rac_features = extend_racs(scaled_features, rac_lig_scaled_features[:,0])
        np.save(base_path+"/"+"extended_rac_features_new_lig_scaled"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], extended_rac_features), axis=1))
        feature_names = "rac_features_depth_3" + "DelP..." 
        np.save(base_path+"/"+"ephys_rac_hybrid_features_new_lig_scaled"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, scaled_features_mcdl46_extended), axis=1))
        feature_names = "rac_features_depth_3" + "DelDB..." 
        #np.save(base_path+"/"+"dbloc_rac_hybrid_features_new_lig_scaled"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, scaled_features_mcdl46_extended_dbloc), axis=1))
        features = np.vstack(rac_features[:, 2])
        scaler = StandardScaler()
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        np.save(base_path+"/"+"rac_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features), axis=1))
        extended_rac_features = extend_racs(scaled_features, rac_lig_scaled_features[:,0])
        np.save(base_path+"/"+"extended_rac_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], extended_rac_features), axis=1))
        np.save(base_path+"/"+"ephys_rac_hybrid_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, scaled_features_mcdl46_extended), axis=1))
        #np.save(base_path+"/"+"dbloc_rac_hybrid_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, scaled_features_mcdl46_extended_dbloc), axis=1))
        features = np.vstack(rac_mean_features[:, 2])
        scaler = StandardScaler()
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        np.save(base_path+"/"+"rac_features_mean_as_ms"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features), axis=1))
        extended_rac_features = extend_racs(scaled_features, rac_lig_scaled_features[:,0])
        np.save(base_path+"/"+"extended_rac_features_mean_as_ms"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], extended_rac_features), axis=1))
        print(scaled_features_mcdl46_extended.shape)
        print(scaled_features.shape)
        
        continue
        #np.save("rac_features_new", rac_features)
        
        features = np.vstack(mcdl46_features[:, 2])
        scaler = StandardScaler()
        scaler.fit(features)
        scaled_features_mcdl46 = scaler.transform(features)
        features_mcdl46 = np.vstack(mcdl46_features[:, 2])
        """
        np.save("mcdl46_features_scaled_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features_mcdl46), axis=1))
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features_mcdl46), axis=1).shape)
        np.save("mcdl46_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], features_mcdl46), axis=1))
        #quit()
        """
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features), axis=1).shape)
        print(np.stack(rac_features[:,2]).shape)
        np.save(base_path+"/"+"hybrid_features_all_scaled_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, scaled_features_mcdl46), axis=1))
        np.save(base_path+"/"+"hybrid_features_new"+restrictions_descriptor, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features, features_mcdl46), axis=1))
        #rac_features_extended = extend_racs(np.stack(rac_features[:,2]), rac_features[:,0])
        """
        rac_features_extended = extend_racs(scaled_features, rac_features[:,0])
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1).shape)
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1))
        print(type(restrictions_descriptor))
        print(restrictions_descriptor)
        print("rac_features_extended_new"+restrictions_descriptor+"_hs_ls_acc_"+hs_ls_accumulated_property)
        np.save("rac_features_extended_new"+restrictions_descriptor+"_hs_ls_acc_"+hs_ls_accumulated_property, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1))
        rac_features_extended = extend_racs(scaled_features, rac_features[:,0], 'phys_ediffs')
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1).shape)
        np.save("rac_features_extended_new_phys_ediff"+restrictions_descriptor+"_hs_ls_acc_"+hs_ls_accumulated_property, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1))
        rac_features_extended = extend_racs(scaled_features, rac_features[:,0], 'dbloc_ediffs')
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1).shape)
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1))
        np.save("rac_features_extended_new_dbloc_ediff"+restrictions_descriptor+"_hs_ls_acc_"+hs_ls_accumulated_property, np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1))
        quit() 
        rac_features_extended = extend_racs(features, rac_features[:,0])
        scaler = StandardScaler()
        scaler.fit(features)
        scaled_features_extended = scaler.transform(features)
        np.save("rac_features_extended_new_standard_scaled", np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features_extended), axis=1))
        print(np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], rac_features_extended), axis=1).shape)
        print(rac_features_extended.shape)
        print(rac_features_non_flat.shape)
        print(rac_features_non_flat[:,2].shape)
        print(rac_features_non_flat[:,2].copy().shape)
        print(np.vstack(rac_features_non_flat[:,2].copy()).shape)
        print(np.stack(rac_features_non_flat[:,2].copy()).shape)
        rac_features_lig_scaled = np.stack(rac_features_non_flat[:,2].copy())
        rac_features_lig_scaled = rac_features_lig_scaled.reshape(-1, rac_features_lig_scaled.shape[1], rac_features_lig_scaled.shape[2]*rac_features_lig_scaled.shape[3])
        rac_features_lig_scaled = get_set_of_lig_scaled_RACs(rac_features_lig_scaled, lig_trackers) 
        print("here", rac_features_lig_scaled.shape)
        rac_features_new_ligand_scaled_extended = extend_racs(rac_features_lig_scaled, rac_features_non_flat[:, 0])
        np.save("rac_features_new_ligand_scaled", np.concatenate([rac_features[:,0][:, np.newaxis], rac_features[:,1][:, np.newaxis], rac_features_lig_scaled], axis=1))
        np.save("rac_features_new_ligand_scaled_extended", np.concatenate([rac_features[:,0][:, np.newaxis], rac_features[:,1][:, np.newaxis], rac_features_new_ligand_scaled_extended], axis=1))
        scaler = StandardScaler()
        scaler.fit(rac_features_new_ligand_scaled_extended)
        scaled_features_extended = scaler.transform(rac_features_new_ligand_scaled_extended)
        np.save("rac_features_new_ligand_scaled_extended_standard_scaled", np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features_extended), axis=1))
        print(np.concatenate([rac_features[:,0][:, np.newaxis], rac_features[:,1][:, np.newaxis], rac_features_lig_scaled], axis=1).shape)
        print("here: ", rac_features_new_ligand_scaled_extended.shape)
        rac_features_lig_scaled = np.stack(rac_features_non_flat[:,2].copy())
        rac_features_lig_scaled = rac_features_lig_scaled.reshape(-1, rac_features_lig_scaled.shape[1], rac_features_lig_scaled.shape[2]*rac_features_lig_scaled.shape[3])
        rac_features_lig_scaled = get_set_of_lig_scaled_RACs(rac_features_lig_scaled, lig_trackers, feature_preserved=True) 
        print("here", rac_features_lig_scaled.shape)
        rac_features_new_ligand_scaled_extended = extend_racs(rac_features_lig_scaled, rac_features_non_flat[:, 0])
        np.save("rac_features_new_ligand_scaled_feature_preserved", np.concatenate([rac_features[:,0][:, np.newaxis], rac_features[:,1][:, np.newaxis], rac_features_lig_scaled], axis=1))
        np.save("rac_features_new_ligand_scaled_extended_feature_preserved", np.concatenate([rac_features[:,0][:, np.newaxis], rac_features[:,1][:, np.newaxis], rac_features_new_ligand_scaled_extended], axis=1))
        scaler = StandardScaler()
        scaler.fit(rac_features_new_ligand_scaled_extended)
        scaled_features_extended = scaler.transform(rac_features_new_ligand_scaled_extended)
        np.save("rac_features_new_ligand_scaled_extended_feature_preserved_standard_scaled", np.concatenate((rac_features[:,0][:, np.newaxis],rac_features[:,1][:, np.newaxis], scaled_features_extended), axis=1))
        #scaled 
        features = np.vstack(rac_features_non_flat[:, 2])
        features = rac_features_lig_scaled
        #scaler = StandardScaler()
        #scaler.fit(features)
        #scaled_features = scaler.transform(features)
        scaled_features = rac_features_new_ligand_scaled_extended
        """
        pca = PCA(n_components=3)
        #hybrid_features = np.hstack([scaled_features, scaled_features_mcdl46])
        #print(hybrid_features.shape)
        principalComponents = pca.fit_transform(scaled_features)
        #principalComponents = pca.fit_transform(hybrid_features)
        #principalComponents = pca.fit_transform(rac_features_extended)
        fig, ax = plt.subplots(figsize=(8,8))

        ax.scatter(principalComponents.T[0], principalComponents.T[1], s=25)
        plt.show()
        print(rac_features[:,0][:, np.newaxis][principalComponents.T[1] > 4][:, 0])
        print(pca.explained_variance_ratio_)
    
    


if __name__ == '__main__':
    main()
