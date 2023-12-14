import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from fcTMCml.featurizer.RAC import RAC
from fcTMCml.featurizer.MCDL53 import MCDL53
from fcTMCml.featurizer.LFF import LigandFieldFeatures
from fcTMCml.featurizer.CFF import CrystalFieldFeatures

from fcTMCml.featurizer.mol_graph_tools import graph_from_xyz_file

from fcTMCml.constants import raw_data_dir, feature_target_dir
from fcTMCml.tools import make_dir, unpack_tar

"""
TODO:
- construct MCDL53 features
- construct MCDL53+CFF features
- construct RAC300 features
- construct RAC300+CFF features
- scale all features
- construct MCDL53+LFF features
- construct RAC300+LFF features
- scale features

"""

# TODO: if geometries does not exist untar geometries.tar.gz

make_dir(feature_target_dir)
unpack_tar("data/geometries")


###########################
# Geometry Classification #
###########################

df = pd.read_csv(raw_data_dir + "thd_geom_classifier.csv")

#  MCDL46
classifier_targets = []
mcdl53_classifier_features = []
mcdl53_cff_classifier_features = []
rac300_classifier_features = []
rac300_cff_classifier_features = []

# ,metal,ox,ligstr,complex.size,spin,geom

for i, row in df.iterrows():
    xyz_file_path = raw_data_dir + f"geometries/metal_{row['metal']}_ox_{int(row['ox'])}_spin_{int(row['spin'])}_ligstr_{row['ligstr']}.xyz"
    if i == 828:  # xyz_file_path == raw_data_dir + "geometries/metal_fe_ox_2_spin_1_ligstr_scn_furan_furan_furan.xyz":
        continue
    ligand_list = row["ligstr"].split("_")
    mcdlf = MCDL53(graph_from_xyz_file(xyz_file_path), oxidation_state=int(row['ox']), ligand_list=ligand_list,
                   multiplicity=row['spin'], input_file=xyz_file_path)
    cff = CrystalFieldFeatures(row["metal"], int(row['ox']), int(row['spin']))
    rac = RAC(graph_from_xyz_file(xyz_file_path), int(row['ox']), ligand_list=ligand_list)
    
    geometry_one_hot = 0 if row["geom"] == 'tetrahedral' else 1
    classifier_targets += [geometry_one_hot]
    f = mcdlf.get_classifier_features()
    fn = mcdlf.get_classifier_feature_names()
    mcdl53_classifier_features += [f]
    f = mcdlf.get_classifier_features(additional_featurizer=[cff])
    mcdl53_cff_classifier_features += [f]
    f = rac.get_classifier_features()
    rac300_classifier_features += [f]
    f = rac.get_classifier_features(additional_featurizer=[cff])
    rac300_cff_classifier_features += [f]
    assert len(f) == len(fn)

mcdl53_classifier_feature_names = mcdlf.get_classifier_feature_names()
mcdl53_cff_classifier_feature_names = mcdlf.get_classifier_feature_names(additional_featurizer=[cff])
rac300_classifier_feature_names = rac.get_classifier_feature_names()
rac300_cff_classifier_feature_names = rac.get_classifier_feature_names(additional_featurizer=[cff])

np.save(feature_target_dir + "classifier_targets.npy", np.array(classifier_targets))

f = np.vstack(mcdl53_classifier_features)
z_scores = StandardScaler()
f = z_scores.fit_transform(f)
np.save(feature_target_dir + "MCDL53_classifier.npy", f)

f = np.vstack(mcdl53_cff_classifier_features)
z_scores = StandardScaler()
f = z_scores.fit_transform(f)
np.save(feature_target_dir + "MCDL53_cff_classifier.npy", f)
f = np.vstack(rac300_classifier_features)
z_scores = StandardScaler()
f = z_scores.fit_transform(f)
np.save(feature_target_dir + "RAC_classifier.npy", f)
f = np.vstack(rac300_cff_classifier_features)
z_scores = StandardScaler()
f = z_scores.fit_transform(f)
np.save(feature_target_dir + "RAC_cff_classifier.npy", f)

quit()


#######################
# SSE regression task #
#######################
# ,metal,ox,ligstr,complex.size,geom.ls,geom.hs,ls.spin,hs.spin,b3lyp.energy.ls (Ha),b3lyp.energy.hs (Ha),b3lyp.sse (kcal/mol)
pd.read_csv(raw_data_dir + "thd_sse_prediction.csv")


#########################################
# Hybrid classification/regression task #
#########################################
# TODO: predict geometry for each state and if state1 != state2, SSE value should be np.nan
