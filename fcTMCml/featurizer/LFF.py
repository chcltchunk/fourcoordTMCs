# implements ligand field features




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


