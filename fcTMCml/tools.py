import importlib.util
import os
import numpy as np


def openbabel_available() -> bool:
    return False
    # checks if openbabel is installed
    openbabel_available = importlib.util.find_spec("openbabel")
    return openbabel_available is not None


def make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def unpack_tar(path: str) -> None:
    if not os.path.exists(path):
        os.system("tar -xcvf " + path + ".tar.gz")


def remove_dir(path: str) -> None:
    os.removedirs(path)


def load_features(feature_target_dir, sub_dir, type="regression") -> dict:
    if type == "classification":
        targets = np.load(feature_target_dir + sub_dir + "classifier_targets.npy")

        mcdl53_features = np.load(feature_target_dir + sub_dir + "MCDL53_classifier.npy")

        mcdl53_cff_features = np.load(feature_target_dir + sub_dir + "MCDL53_cff_classifier.npy")

        rac300_features = np.load(feature_target_dir + sub_dir + "RAC_classifier.npy")

        rac300_cff_features = np.load(feature_target_dir + sub_dir + "RAC_cff_classifier.npy")
    elif type == "regression":
        targets = np.load(feature_target_dir + sub_dir + "regression_targets.npy")

        mcdl53_features = np.load(feature_target_dir + sub_dir + "MCDL53_regression.npy")

        mcdl53_cff_features = np.load(feature_target_dir + sub_dir + "MCDL53_cff_regression.npy")

        rac300_features = np.load(feature_target_dir + sub_dir + "RAC_regression.npy")

        rac300_cff_features = np.load(feature_target_dir + sub_dir + "RAC_cff_regression.npy")
    else:
        raise TypeError("feature type not supported")
    return {f"{type}_targets" : targets,
            f"mcdl53_{type}_features" : mcdl53_features,
            f"mcdl53_cff_{type}_features" : mcdl53_cff_features,
            f"rac300_{type}_features" : rac300_features,
            f"rac300_cff_{type}_features" : rac300_cff_features
            }
