import importlib.util
import os
import numpy as np


from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    return targets, {f"mcdl53_{type}_features" : mcdl53_features,
                     f"mcdl53_cff_{type}_features" : mcdl53_cff_features,
                     f"rac300_{type}_features" : rac300_features,
                     f"rac300_cff_{type}_features" : rac300_cff_features
                     }


def get_pca(features: np.array) -> (list, list):
    pca = PCA()
    principalComponents = pca.fit_transform(features)
    explained_variances = pca.explained_variance_ratio_
    return (principalComponents, explained_variances)


def plot_pca(principalComponents, explained_variance, color_list, filename, color_dic=None, mapper=None, legends=None, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(principalComponents.T[0], principalComponents.T[1], s=25, c=color_list)
    ax.set_xlabel("PC 1 ({})".format(np.round(explained_variance[0], 2)), fontsize=20)
    ax.set_ylabel("PC 2 ({})".format(np.round(explained_variance[1], 2)), fontsize=20)
    ax.tick_params(which="major", direction="in")

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if color_dic is not None:
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_dic.values()]
        plt.legend(markers, color_dic.keys(), numpoints=1, prop={'size': 15})
    elif mapper is not None:
        cbar = plt.colorbar(mapper[0], ticks=np.arange(-70, 11, 10))
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(mapper[1], rotation=90, fontsize=15)
    if legends is not None:
        handles, labels = ax.get_legend_handles_labels()
        patch1 = mpatches.Patch(color="tab:blue", label="THD")
        patch2 = mpatches.Patch(color="tab:orange", label="SQP")
        handles.extend([patch1, patch2])
        ax.legend(handles=handles, fontsize=15)
    if title is not None:
        plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
