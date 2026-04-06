import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from fcTMCml.featurizer.RAC import RAC
from fcTMCml.featurizer.MCDLF import MCDLF
# TODO(jonas): add: from fcTMCml.featurizer.LFF import LigandFieldFeatures
from fcTMCml.featurizer.CFF import CrystalFieldFeatures

from fcTMCml.featurizer.mol_graph_tools import graph_from_xyz_file

from fcTMCml.constants import raw_data_dir, feature_target_dir
from fcTMCml.tools import make_dir, unpack_tar

# TODO: if geometries does not exist untar geometries.tar.gz

# list of RACs that have a trivial value
zero_RACs = ['mc-T-0-all', 'mc-I-0-all', 'mc-I-1-all', 'D_mc-Z-0-all',
             'D_mc-chi-0-all', 'D_mc-T-0-all', 'D_mc-I-0-all', 'D_mc-S-0-all',
             'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'lc-I-0-ax1',
             'lc-I-0-ax2', 'lc-I-0-ax3', 'lc-I-0-ax4', 'D_lc-Z-0-ax1',
             'D_lc-chi-0-ax1', 'D_lc-T-0-ax1', 'D_lc-I-0-ax1', 'D_lc-S-0-ax1',
             'D_lc-I-1-ax1', 'D_lc-I-2-ax1', 'D_lc-I-3-ax1', 'D_lc-Z-0-ax2',
             'D_lc-chi-0-ax2', 'D_lc-T-0-ax2', 'D_lc-I-0-ax2', 'D_lc-S-0-ax2',
             'D_lc-I-1-ax2', 'D_lc-I-2-ax2', 'D_lc-I-3-ax2', 'D_lc-Z-0-ax3',
             'D_lc-chi-0-ax3', 'D_lc-T-0-ax3', 'D_lc-I-0-ax3', 'D_lc-S-0-ax3',
             'D_lc-I-1-ax3', 'D_lc-I-2-ax3', 'D_lc-I-3-ax3', 'D_lc-Z-0-ax4',
             'D_lc-chi-0-ax4', 'D_lc-T-0-ax4', 'D_lc-I-0-ax4', 'D_lc-S-0-ax4',
             'D_lc-I-1-ax4', 'D_lc-I-2-ax4', 'D_lc-I-3-ax4']

make_dir(feature_target_dir)
classification_subdir = "classification_raw/"
regression_subdir = "regression_raw/"
make_dir(feature_target_dir + classification_subdir)
make_dir(feature_target_dir + regression_subdir)
unpack_tar("data/geometries")


###########################
# Geometry Classification #
###########################

df = pd.read_csv(raw_data_dir + "thd_geom_classifier.csv")

classifier_targets = []
classifier_target_names = []
mcdlf_classifier_features = []
mcdlf_cff_classifier_features = []
rac_classifier_features = []
rac_cff_classifier_features = []

for i, row in df.iterrows():
    xyz_file_path = raw_data_dir + f"geometries/metal_{row['metal']}_ox_{int(row['ox'])}_spin_{int(row['spin'])}_ligstr_{row['ligstr']}.xyz"
    if i == 828:  # xyz_file_path == raw_data_dir + "geometries/metal_fe_ox_2_spin_1_ligstr_scn_furan_furan_furan.xyz":
        continue
    ligand_list = row["ligstr"].split("_")
    mcdlf = MCDLF(graph_from_xyz_file(xyz_file_path), oxidation_state=int(row['ox']), ligand_list=ligand_list,
                  multiplicity=row['spin'], input_file=xyz_file_path)
    cff = CrystalFieldFeatures(row["metal"], int(row['ox']), int(row['spin']))
    rac = RAC(graph_from_xyz_file(xyz_file_path), int(row['ox']), ligand_list=ligand_list)

    geometry_one_hot = 0 if row["geom"] == 'tetrahedral' else 1
    classifier_targets += [geometry_one_hot]
    classifier_target_names += [xyz_file_path.split("/")[-1].split(".")[0]]
    f = mcdlf.get_classifier_features()
    mcdlf_classifier_features += [f]
    f = mcdlf.get_classifier_features(additional_featurizer=[cff])
    mcdlf_cff_classifier_features += [f]
    f = rac.get_classifier_features()
    rac_classifier_features += [f]
    f = rac.get_classifier_features(additional_featurizer=[cff])
    rac_cff_classifier_features += [f]

mcdlf_classifier_feature_names = mcdlf.get_classifier_feature_names()
mcdlf_classifier_feature_groups = mcdlf.get_classifier_feature_groups()
assert len(mcdlf_classifier_features[0]) == len(mcdlf_classifier_feature_names)
mcdlf_cff_classifier_feature_names = mcdlf.get_classifier_feature_names(additional_featurizer=[cff])
mcdlf_cff_classifier_feature_groups = mcdlf.get_classifier_feature_groups(additional_featurizer=[cff])
assert len(mcdlf_cff_classifier_features[0]) == len(mcdlf_cff_classifier_feature_names)
rac_classifier_feature_names = rac.get_classifier_feature_names()
rac_classifier_feature_groups = rac.get_classifier_feature_groups()
# select non-trivial values only
mask = ~np.isin(rac_classifier_feature_names, zero_RACs)
rac_classifier_feature_names = [names for i, names in enumerate(rac_classifier_feature_names) if mask[i]] #rac_classifier_feature_names[mask]
rac_classifier_feature_groups = [groups for i, groups in enumerate(rac_classifier_feature_groups) if mask[i]]
rac_classifier_features = [[rac for rac, m in zip(row, mask) if m]
                           for row in rac_classifier_features]
assert len(rac_classifier_features[0]) == len(rac_classifier_feature_names)
rac_cff_classifier_feature_names = rac.get_classifier_feature_names(additional_featurizer=[cff])
rac_cff_classifier_feature_groups = rac.get_classifier_feature_groups(additional_featurizer=[cff])
# select non-trivial values only
mask = ~np.isin(rac_cff_classifier_feature_names, zero_RACs)
rac_cff_classifier_feature_names = [names for i, names in enumerate(rac_cff_classifier_feature_names) if mask[i]] #rac_classifier_feature_names[mask]
rac_cff_classifier_feature_groups = [groups for i, groups in enumerate(rac_cff_classifier_feature_groups) if mask[i]]
rac_cff_classifier_features = [[rac for rac, m in zip(row, mask) if m]
                               for row in rac_cff_classifier_features]
assert len(rac_cff_classifier_features[0]) == len(rac_cff_classifier_feature_names)

np.save(feature_target_dir + classification_subdir + "classification_targets.npy", classifier_targets)

np.save(feature_target_dir + classification_subdir + "classification_target_names.npy", classifier_target_names)

feature_variants = {'MCDLF_classification' : [mcdlf_classifier_features, mcdlf_classifier_feature_names, mcdlf_classifier_feature_groups],
                    'MCDLF_cff_classification': [mcdlf_cff_classifier_features, mcdlf_cff_classifier_feature_names, mcdlf_cff_classifier_feature_groups],
                    'RAC_classification': [rac_classifier_features, rac_classifier_feature_names, rac_classifier_feature_groups],
                    'RAC_cff_classification': [rac_cff_classifier_features, rac_cff_classifier_feature_names, rac_cff_classifier_feature_groups]
                    }

for feature_variant in feature_variants:
    f = np.vstack(feature_variants[feature_variant][0])
    z_scores = StandardScaler()
    f = z_scores.fit_transform(f)  # X array-like of shape (n_samples, n_features)
    np.save(feature_target_dir + classification_subdir + f"{feature_variant}.npy", f)
    np.save(feature_target_dir + classification_subdir + f"{feature_variant}_names.npy", feature_variants[feature_variant][1])
    np.save(feature_target_dir + classification_subdir + f"{feature_variant}_groups.npy", feature_variants[feature_variant][2])


#######################
# SSE regression task #
#######################
# ,metal,ox,ligstr,complex.size,geom.ls,geom.hs,ls.spin,hs.spin,b3lyp.energy.ls (Ha),b3lyp.energy.hs (Ha),b3lyp.sse (kcal/mol)
df = pd.read_csv(raw_data_dir + "thd_sse_prediction.csv")

regression_targets = []
regression_target_names = []
mcdlf_regression_features = []
mcdlf_cff_regression_features = []
rac_regression_features = []
rac_cff_regression_features = []

for i, row in df.iterrows():
    xyz_file_path = raw_data_dir + f"geometries/metal_{row['metal']}_ox_{int(row['ox'])}_spin_{int(row['ls.spin'])}_ligstr_{row['ligstr']}.xyz"
    if i == 828:  # xyz_file_path == raw_data_dir + "geometries/metal_fe_ox_2_spin_1_ligstr_scn_furan_furan_furan.xyz":
        continue
    ligand_list = row["ligstr"].split("_")
    mcdlf = MCDLF(graph_from_xyz_file(xyz_file_path), oxidation_state=int(row['ox']), ligand_list=ligand_list,
                  multiplicity=None, input_file=xyz_file_path)
    cff = CrystalFieldFeatures(row["metal"], int(row['ox']), [int(row['ls.spin']), int(row['hs.spin'])], geometryA='tetrahedral', geometryB='tetrahedral')
    rac = RAC(graph_from_xyz_file(xyz_file_path), int(row['ox']), ligand_list=ligand_list)

    regression_targets += [row["b3lyp.sse (kcal/mol)"]]
    regression_target_names += [xyz_file_path.split("/")[-1].split(".")[0]]
    f = mcdlf.get_regression_features()
    mcdlf_regression_features += [f]
    f = mcdlf.get_regression_features(additional_featurizer=[cff])
    mcdlf_cff_regression_features += [f]
    f = rac.get_regression_features()
    rac_regression_features += [f]
    f = rac.get_regression_features(additional_featurizer=[cff])
    rac_cff_regression_features += [f]

mcdlf_regression_feature_names = mcdlf.get_regression_feature_names()
mcdlf_regression_feature_groups = mcdlf.get_regression_feature_groups()
assert len(mcdlf_regression_features[0]) == len(mcdlf_regression_feature_names)
mcdlf_cff_regression_feature_names = mcdlf.get_regression_feature_names(additional_featurizer=[cff])
mcdlf_cff_regression_feature_groups = mcdlf.get_regression_feature_groups(additional_featurizer=[cff])
assert len(mcdlf_cff_regression_features[0]) == len(mcdlf_cff_regression_feature_names)
rac_regression_feature_names = rac.get_regression_feature_names()
rac_regression_feature_groups = rac.get_regression_feature_groups()
assert len(rac_regression_features[0]) == len(rac_regression_feature_names)
rac_cff_regression_feature_names = rac.get_regression_feature_names(additional_featurizer=[cff])
rac_cff_regression_feature_groups = rac.get_regression_feature_groups(additional_featurizer=[cff])
assert len(rac_cff_regression_features[0]) == len(rac_cff_regression_feature_names)

np.save(feature_target_dir + regression_subdir + "regression_targets.npy", np.array(regression_targets))

np.save(feature_target_dir + regression_subdir + "regression_target_names.npy", np.array(regression_target_names))

feature_variants = {'MCDLF_regression' : [mcdlf_regression_features, mcdlf_regression_feature_names, mcdlf_regression_feature_groups],
                    'MCDLF_cff_regression': [mcdlf_cff_regression_features, mcdlf_cff_regression_feature_names, mcdlf_cff_regression_feature_groups],
                    'RAC_regression': [rac_regression_features, rac_regression_feature_names, rac_regression_feature_groups],
                    'RAC_cff_regression': [rac_cff_regression_features, rac_cff_regression_feature_names, rac_cff_regression_feature_groups]
                    }

for feature_variant in feature_variants:
    f = np.vstack(feature_variants[feature_variant][0])
    z_scores = StandardScaler()
    f = z_scores.fit_transform(f)  # X array-like of shape (n_samples, n_features)
    np.save(feature_target_dir + regression_subdir + f"{feature_variant}.npy", f)
    np.save(feature_target_dir + regression_subdir + f"{feature_variant}_names.npy", feature_variants[feature_variant][1])
    np.save(feature_target_dir + regression_subdir + f"{feature_variant}_groups.npy", feature_variants[feature_variant][2])
