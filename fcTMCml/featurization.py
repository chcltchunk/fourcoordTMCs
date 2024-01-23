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

"""
TODO:
- construct MCDLF+LFF features
- construct RAC+LFF features
"""

# TODO: if geometries does not exist untar geometries.tar.gz


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
assert len(rac_classifier_features[0]) == len(rac_classifier_feature_names)
rac_cff_classifier_feature_names = rac.get_classifier_feature_names(additional_featurizer=[cff])
rac_cff_classifier_feature_groups = rac.get_classifier_feature_groups(additional_featurizer=[cff])
assert len(rac_cff_classifier_features[0]) == len(rac_cff_classifier_feature_names)

np.save(feature_target_dir + classification_subdir + "classification_targets.npy", classifier_targets)

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

#########################################
# Hybrid classification/regression task #
#########################################
# TODO: predict geometry for each state and if state1 != state2, SSE value should be np.nan
