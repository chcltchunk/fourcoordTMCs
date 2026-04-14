import numpy as np
from ase.data import chemical_symbols, atomic_numbers, covalent_radii


# ligand dict parsing
def load_ligand_dict(path: str) -> dict:
    # load molSimplify ligand dict from ligands.dict
    with open(path, "r") as f:
        lines = f.readlines()
    return {x.split(":")[0]: x.split(":")[1][:-1].split(",") for x in lines[2:]}


##############################
# PROJECT SPECIFIC CONSTANTS #
##############################

# in DOI: 10.1039/c7sc01247k cutoff is <= 1 (gives 242 less datapoints)
S2_CUTOFF = 1.5
# SSE lower than -110 kcal/mol is not reasonable for this dataset
SSE_CUTOFF = -110

# floating point accuracy for tests
EPS = 1e-6
test_resource_dir = "tests/testresources/"
raw_data_dir = "data/"
feature_target_dir = "fcTMCml/features/"
cache_dir = "cache/"
ligand_dict_dir = "fcTMCml/featurizer/ligands.dict"
SHAP_directory = "results/feature_importance_analysis/SHAP/"

ligand_dict = load_ligand_dict(ligand_dict_dir)

# define properties we'd like to keep in the dataset throughout the complete workflow
column_list = ['metal', 'ox', 'ligstr', 'complex.size']  # , 'charge'
sse_colum_list = ['geom.ls', 'geom.hs', 'ls.spin', 'hs.spin', 'b3lyp.energy.ls (Ha)', 'b3lyp.energy.hs (Ha)', 'b3lyp.sse (kcal/mol)']

#####################
# GENERAL CONSTANTS #
#####################

__all__ = [
    "chemical_symbols",
    "atomic_numbers",
    "atomic_masses",
    "covalent_radii",
]

roman_numerals = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
roman_numerals_r = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

metal_list_symbolic = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
metal_list = np.array(list(range(21, 30)) + list(range(39, 48)) + list(range(72, 80)))

# Consistent with the definition in molSimplify:
# http://www.webelements.com/ (last accessed May 13th 2015)
atomic_masses = np.array(
    [
        1.0,  # X
        1.0079,  # H
        4.002602,  # He
        6.94,  # Li
        9.0121831,  # Be
        10.83,  # B
        12.0107,  # C
        14.0067,  # N
        15.9994,  # O
        18.9984,  # F
        20.1797,  # Ne
        22.99,  # Na
        24.3,  # Mg
        26.98,  # Al
        28.08,  # Si
        30.9738,  # P
        32.065,  # S
        35.453,  # Cl
        39.948,  # Ar
        39.1,  # K
        40.08,  # Ca
        44.96,  # Sc
        47.867,  # Ti
        50.94,  # V
        51.9961,  # Cr
        54.938,  # Mn
        55.84526,  # Fe
        58.9332,  # Co
        58.4934,  # Ni
        63.546,  # Cu
        65.39,  # Zn
        69.72,  # Ga
        72.63,  # Ge
        74.92,  # As
        78.96,  # Se
        79.904,  # Br
        83.798,  # Kr
        85.47,  # Rb
        87.62,  # Sr
        88.91,  # Y
        91.22,  # Zr
        92.91,  # Nb
        95.96,  # Mo
        98.9,  # Tc
        101.1,  # Ru
        102.9,  # Rh
        106.4,  # Pd
        107.9,  # Ag
        112.4,  # Cd
        111.818,  # In
        118.71,  # Sn
        121.76,  # Sb
        127.6,  # Te
        126.90447,  # I
        131.293,  # Xe
        132.9055,  # Cs
        137.327,  # Ba
        138.9,  # La
        140.116,  # Ce
        140.90766,  # Pr
        144.242,  # Nd
        145,  # Pm
        150.36,  # Sm
        151.964,  # Eu
        157.25,  # Gd
        158.92535,  # Tb
        162.5,  # Dy
        164.93033,  # Ho
        167.259,  # Er
        168.93422,  # Tm
        173.045,  # Yb
        174.9668,  # Lu
        178.5,  # Hf
        180.9,  # Ta
        183.8,  # W
        186.2,  # Re
        190.2,  # Os
        192.2,  # Ir
        195.1,  # Pt
        197.0,  # Au
        200.6,  # Hg
        204.38,  # Tl
        207.2,  # Pb
        208.9804,  # Bi
        208.98,  # Po
        209.99,  # At
        222.6,  # Rn
        223.02,  # Fr
        226.03,  # Ra
        277,  # Ac
        232.0377,  # Th
        231.04,  # Pa
        238.02891,  # U
        237.05,  # Np
        244.06,  # Pu
        243.06,  # Am
        247.07,  # Cm
        247.07,  # Bk
        251.08,  # Cf
    ]
)

# From:
# https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)
no_data = np.NaN
electronegativity = np.array(
    [
        no_data,  # X
        2.20,  # H
        no_data,  # He
        0.98,  # Li
        1.57,  # Be
        2.04,  # B
        2.55,  # C
        3.04,  # N
        3.44,  # O
        3.98,  # F
        no_data,  # Ne
        0.93,  # Na
        1.31,  # Mg
        1.61,  # Al
        1.90,  # Si
        2.19,  # P
        2.58,  # S
        3.16,  # Cl
        no_data,  # Ar
        0.82,  # K
        1.00,  # Ca
        1.36,  # Sc
        1.54,  # Ti
        1.63,  # V
        1.66,  # Cr
        1.55,  # Mn
        1.83,  # Fe
        1.88,  # Co
        1.91,  # Ni
        1.90,  # Cu
        1.65,  # Zn
        1.81,  # Ga
        2.01,  # Ge
        2.18,  # As
        2.55,  # Se
        2.96,  # Br
        3.00,  # Kr
        0.82,  # Rb
        0.95,  # Sr
        1.22,  # Y
        1.33,  # Zr
        1.60,  # Nb
        2.16,  # Mo
        1.90,  # Tc
        2.20,  # Ru
        2.28,  # Rh
        2.20,  # Pd
        1.93,  # Ag
        1.69,  # Cd
        1.78,  # In
        1.96,  # Sn
        2.05,  # Sb
        2.10,  # Te
        2.66,  # I
        2.60,  # Xe
        0.79,  # Cs
        0.89,  # Ba
        1.10,  # La
        1.12,  # Ce
        1.13,  # Pr
        1.14,  # Nd
        no_data,  # Pm
        1.17,  # Sm
        no_data,  # Eu
        1.20,  # Gd
        no_data,  # Tb
        1.22,  # Dy
        1.23,  # Ho
        1.24,  # Er
        1.25,  # Tm
        no_data,  # Yb
        1.27,  # Lu
        1.30,  # Hf
        1.50,  # Ta
        2.36,  # W
        1.90,  # Re
        2.20,  # Os
        2.20,  # Ir
        2.28,  # Pt
        2.54,  # Au
        2.00,  # Hg
        1.62,  # Tl
        2.33,  # Pb
        2.02,  # Bi
        2.00,  # Po
        2.20,  # At
        no_data,  # Rn
        no_data,  # Fr
        0.90,  # Ra
        1.10,  # Ac
        1.30,  # Th
        1.50,  # Pa
        1.38,  # U
        1.36,  # Np
        1.28,  # Pu
        1.30,  # Am
        1.30,  # Cm
        1.30,  # Bk
        1.30,  # Cf
        1.30,  # Es
        1.30,  # Fm
        1.30,  # Md
        1.30,  # No
    ]
)
