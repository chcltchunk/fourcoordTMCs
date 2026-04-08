import importlib.util
import os
import numpy as np

from scipy.stats import gaussian_kde

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit


from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial.distance import mahalanobis
from scipy.linalg import sqrtm

from typing import Union

from fcTMCml.constants import roman_numerals_r


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


def plot_geometry_convergence_overview(dataframe, filename):
    fig, ax = plt.subplots(2, 9, figsize=(40, 10))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, sc in enumerate(["geom.ls", "geom.hs"]):
        j = -1
        for metal in ["cr", "mn", "fe", "co", "ni"]:
            for ox in dataframe[dataframe["metal"] == metal]["ox"].unique():
                j += 1
                mask = (dataframe["metal"] == metal) & (dataframe["ox"] == ox)
                thd_count = np.count_nonzero((dataframe[sc] == "tetrahedral") & mask)
                sqp_count = np.count_nonzero((dataframe[sc] == "square planar") & mask)
                other_count = len(dataframe[mask].to_numpy()) - thd_count - sqp_count
                ax[i, j].set_aspect("equal")
                ax[i, j].pie([thd_count, sqp_count, other_count],  # labels=["THD", "SQP", "other"],
                             colors=['dodgerblue', 'tab:orange', 'gray'], labeldistance=.4, shadow=True, wedgeprops=dict(width=0.6), startangle=-40)
                if i == 0:
                    ax[i, j].set_title(f"{metal.capitalize()} ({roman_numerals_r[int(ox)]})", fontsize=40)
                    if j == 0:
                        ax[i, j].set_ylabel("low spin", fontsize=40)
                elif j == 0:
                    ax[i, j].set_ylabel("high spin", fontsize=40)
        handles = [Line2D([0], [0], marker='o', color='w', label='THD',
                   markerfacecolor='dodgerblue', markersize=35, alpha=0.8),
                   Line2D([0], [0], marker='o', color='w', label='SQP',
                   markerfacecolor='tab:orange', markersize=35, alpha=0.8),
                   Line2D([0], [0], marker='o', color='w', label='other',
                   markerfacecolor='grey', markersize=35, alpha=0.8)
                   ]
        fig.legend(handles, ["THD", "SQP", "other"], fontsize=35, loc=7)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def load_features(feature_target_dir, sub_dir, type="regression", load_groups: bool = False, load_importances: bool = False) -> dict:
    targets = np.load(feature_target_dir + sub_dir + f"{type}_targets.npy")

    mcdlf_features = np.load(feature_target_dir + sub_dir + f"MCDLF_{type}.npy")
    mcdlf_feature_names = np.load(feature_target_dir + sub_dir + f"MCDLF_{type}_names.npy")

    mcdlf_cff_features = np.load(feature_target_dir + sub_dir + f"MCDLF_cff_{type}.npy")
    mcdlf_cff_feature_names = np.load(feature_target_dir + sub_dir + f"MCDLF_cff_{type}_names.npy")

    rac_features = np.load(feature_target_dir + sub_dir + f"RAC_{type}.npy")
    rac_feature_names = np.load(feature_target_dir + sub_dir + f"RAC_{type}_names.npy")

    rac_cff_features = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}.npy")
    rac_cff_feature_names = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}_names.npy")
    return_dictionaries = [{f"MCDLF_{type}" : mcdlf_features,
                            f"MCDLF_cff_{type}" : mcdlf_cff_features,
                            f"RAC_{type}" : rac_features,
                            f"RAC_cff_{type}" : rac_cff_features
                            }, {f"MCDLF_{type}" : mcdlf_feature_names,
                                f"MCDLF_cff_{type}" : mcdlf_cff_feature_names,
                                f"RAC_{type}" : rac_feature_names,
                                f"RAC_cff_{type}" : rac_cff_feature_names
                                }]
    if load_groups:
        mcdlf_feature_groups = np.load(feature_target_dir + sub_dir + f"MCDLF_{type}_groups.npy")
        mcdlf_cff_feature_groups = np.load(feature_target_dir + sub_dir + f"MCDLF_cff_{type}_groups.npy")
        rac_feature_groups = np.load(feature_target_dir + sub_dir + f"RAC_{type}_groups.npy")
        rac_cff_feature_groups = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}_groups.npy")
        return_dictionaries += [{f"MCDLF_{type}" : mcdlf_feature_groups,
                                 f"MCDLF_cff_{type}" : mcdlf_cff_feature_groups,
                                 f"RAC_{type}" : rac_feature_groups,
                                 f"RAC_cff_{type}" : rac_cff_feature_groups
                                 }]
    if load_importances:
        mcdlf_feature_importances = np.load(feature_target_dir + sub_dir + f"MCDLF_{type}_importances.npy")
        mcdlf_cff_feature_importances = np.load(feature_target_dir + sub_dir + f"MCDLF_cff_{type}_importances.npy")
        rac_feature_importances = np.load(feature_target_dir + sub_dir + f"RAC_{type}_importances.npy")
        rac_cff_feature_importances = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}_importances.npy")
        return_dictionaries += [{f"MCDLF_{type}" : mcdlf_feature_importances,
                                 f"MCDLF_cff_{type}" : mcdlf_cff_feature_importances,
                                 f"RAC_{type}" : rac_feature_importances,
                                 f"RAC_cff_{type}" : rac_cff_feature_importances
                                 }]
    return targets, *return_dictionaries


def mahalanobis_dist(X1, X2, regularization=1e-6):
    """
    calculate mahalanobis distance between 2 datasets
    Parameters
    ----------
    X1, X2 : np.ndarray of shape (n_samples, 2)
        Input datasets
    regularization : float
        Small value added to covariance diagonals for stability
    """
    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
    print(mu1, mu2)
    cov1 = np.cov(X1, rowvar=False) + regularization * np.eye(2)
    cov2 = np.cov(X2, rowvar=False) + regularization * np.eye(2)
    cov12 = (cov1 + cov2) / 2
    print(cov12)
    return mahalanobis(mu1, mu2, np.linalg.inv(cov12))


def wasserstein_dist(X1, X2, regularization=1e-6):
    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
    mean_diff = np.linalg.norm(mu1 - mu2) ** 2
    cov1 = np.cov(X1, rowvar=False) + regularization * np.eye(2)
    cov2 = np.cov(X2, rowvar=False) + regularization * np.eye(2)
    cov1_sqrt = sqrtm(cov1)
    middle = cov1_sqrt @ cov2 @ cov1_sqrt
    cov_term = np.real(np.trace(cov1 + cov2 - 2 * sqrtm(middle)))
    return np.sqrt(mean_diff + cov_term)


def get_pca(features: np.array) -> (Union[list, list]):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    explained_variances = pca.explained_variance_ratio_
    return (principalComponents, explained_variances)


def get_tsne(features: np.array, perplexity: float = 30.0) -> list:
    tsne = TSNE(perplexity=perplexity, random_state=256)
    tsne_embedding = tsne.fit_transform(features)
    return tsne_embedding


def get_umap(features: np.array) -> list:
    umap = UMAP()
    umap_embedding = umap.fit_transform(features)
    return umap_embedding


def plot_pca(principalComponents: np.array, explained_variance: np.array , color_list: list, filename: str,
             color_dic: dict = None, mapper=None,
             legends=None, title=None, x_label=None, y_label=None, classification: bool = True):
    fig, ax = plt.subplots(figsize=(8, 8))

    if classification:
        color_list_num = np.where(color_list == "dodgerblue", 0, 1)
        principalComponents_class0 = principalComponents[color_list_num == 0]
        principalComponents_class1 = principalComponents[color_list_num == 1]
        ax.scatter(principalComponents_class0.T[0], principalComponents_class0.T[1], s=80, c="dodgerblue", alpha=0.6, edgecolors='none')
        ax.scatter(principalComponents_class1.T[0], principalComponents_class1.T[1], s=80, c="tab:orange", alpha=0.6, edgecolors='none')
    else:
        ax.scatter(principalComponents.T[0], principalComponents.T[1], s=80, c=color_list, alpha=0.6, edgecolors='none')

    x_label = "PC 1 ({})".format(np.round(explained_variance[0], 2)) if x_label is None else x_label
    y_label = "PC 2 ({})".format(np.round(explained_variance[1], 2)) if y_label is None else y_label
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(which="major", direction="in")

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if color_dic is not None:
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_dic.values()]
        ax.legend(markers, color_dic.keys(), numpoints=1, prop={'size': 15})
    elif mapper is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mapper[0], ticks=np.arange(-110, 11, 10), cax=cax)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(mapper[1], rotation=90, fontsize=15)
    if legends is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([Line2D([0], [0], marker='o', color='w', label='THD',
                        markerfacecolor='dodgerblue', markersize=15, alpha=0.8),
                        Line2D([0], [0], marker='o', color='w', label='SQP',
                        markerfacecolor='tab:orange', markersize=15, alpha=0.8)
                        ])
        ax.legend(handles=handles, fontsize=15)
    if title is not None:
        ax.set_title(title)
    if classification:
        cmap = colors.ListedColormap(['dodgerblue', 'tab:orange'])

        x_min, x_max = principalComponents.T[0].min() - 1, principalComponents.T[0].max() + 1
        y_min, y_max = principalComponents.T[1].min() - 1, principalComponents.T[1].max() + 1

        clf = KNeighborsClassifier(5)
        clf.fit(principalComponents, color_list_num)

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                             np.linspace(y_min, y_max, 1000))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], cmap=cmap, alpha=0.3, aspect='auto', origin='lower')

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tsne(tsne_embedding, color_list, filename, color_dic=None, mapper=None, legends=None, title=None, classification: bool = True):
    plot_pca(tsne_embedding, [0, 0], color_list, filename, color_dic, mapper, legends,
             title, x_label="tSNE-1", y_label="tSNE-2", classification=classification)


def plot_umap(tsne_embedding, color_list, filename, color_dic=None, mapper=None, legends=None, title=None, classification: bool = True):
    plot_pca(tsne_embedding, [0, 0], color_list, filename, color_dic, mapper, legends,
             title, x_label="UMAP-1", y_label="UMAP-2", classification=classification)


def plot_AUC(fig, ax, tprs, aucs, mean_fpr, filename):
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=fr"$\mu$(ROC$_i$) (AUC = {mean_auc:.2f})",
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = mean_tpr + std_tpr  # np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = mean_tpr - std_tpr  # np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ $\sigma$(ROC$_i$)",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    handles, legends = ax.get_legend_handles_labels()
    legends[-4] = r"ROC $i$th-fold"
    handles = handles[-4:]
    legends = legends[-4:]
    # remove chance label
    del legends[1]
    del handles[1]
    ax.legend(handles, legends , loc="lower center")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.tick_params(direction="in")

    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_classifier_probabilities(probs, probs_ref, filename, clasifier_name):
    fig, ax = plt.subplots()
    probs_ref = np.array(probs_ref)
    probs = np.array(probs)
    mask = probs_ref == 0
    probs_ref_thd = np.array(probs_ref[mask]).astype('float64')
    probs_thd = np.array(probs[mask]).astype('float64')
    mask = probs_ref == 1
    probs_ref_sqp = np.array(probs_ref[mask]).astype('float64')
    probs_sqp = np.array(probs[mask]).astype('float64')
    probs_ref_thd += np.random.uniform(-0.5, 0.5, size=probs_ref_thd.shape)
    probs_ref_sqp += np.random.uniform(-0.5, 0.5, size=probs_ref_sqp.shape)
    ax.scatter(probs_ref_thd, probs_thd, edgecolors='none', alpha=0.6, c="dodgerblue")
    ax.scatter(probs_ref_sqp, probs_sqp, edgecolors='none', alpha=0.6, c="tab:orange")
    ax.set_xticks([0, 1], ["THD", "SQP"])
    ax.set_ylabel(f"{clasifier_name} probabilities")
    ax.tick_params(direction="in")

    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, filename: str):
    fig, ax = plt.subplots()

    cax = ax.matshow(confusion_matrix, cmap="Reds", vmin=0, vmax=np.max(confusion_matrix))
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{int(z)}', ha='center', va='center')
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(cax, )
    ax.set_xticks([0, 1], ['THD', 'SQP'])
    ax.set_yticks([0, 1], ['THD', 'SQP'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ax.tick_params(direction="in")

    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax2.set_xticks([0, 1])
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)
    ax3.set_yticks([0, 1])

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curve(X: np.ndarray, y: np.ndarray, estimator, filename: str, clf_name: str):
    fig, ax = plt.subplots(figsize=(5, 6))
    params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.1, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o", "color" : "green"},
        "fill_between_kw": {"color" : "green", "edgecolor" : "none"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
        "random_state": 128,
    }

    LearningCurveDisplay.from_estimator(estimator, **params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.tick_params(direction="in")
    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax2.set_xticks(ax.get_xticks(), labels=[])
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)
    ax2.set_yticks(ax.get_yticks(), labels=[])
    ax.set_ylim(0.65, 1.015)
    ax.lines[0].set_color("blue")
    ax.lines[0].set_linewidth(3)
    ax.lines[1].set_linewidth(3)
    ax.collections[0].set_color("blue")
    ax.collections[0].set_edgecolor(None)
    ax.set_xlabel("Iterations", fontsize=18)
    ax.set_ylabel("MAE / kcal mol$^{-1}$", fontsize=18)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('blue')
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_parity_plot(y_predict_n: list, y_truth_n: list, x_label: str, y_label: str, filename: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(np.min(y_truth_n), np.max(y_truth_n))
    ax.set_ylim(np.min(y_predict_n), np.max(y_predict_n))
    y_truth_n = np.array(y_truth_n).ravel()
    y_predict_n = np.array(y_predict_n).ravel()
    xy = np.vstack([y_truth_n, y_predict_n])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_truth_n[idx], y_predict_n[idx], z[idx]
    scatter = ax.scatter(x, y, c=z, cmap=plt.cm.get_cmap('winter'), s=50)
    ax.set_xlim(-90, 10)
    ax.set_ylim(-90, 10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot([-90, 10], [-90, 10], linestyle="--", lw=2, color="k")
    ax.tick_params(direction="in")

    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)

    cb = plt.colorbar(scatter, )
    cb.ax.tick_params(axis='y', direction='in')
    cb.ax.set_ylabel("Gaussian KDE", rotation=-90, va="bottom")
    cb.ax.set_yticks([])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curve_NN(train_loss, val_loss, filename, sqrt: bool = False):
    fig, ax = plt.subplots(figsize=(5, 6))
    epoch = np.arange(len(train_loss))
    if sqrt:
        train_loss = np.sqrt(train_loss)
        val_loss = np.sqrt(val_loss)
    ax.plot(epoch, train_loss, c="tab:blue", label="Training", lw=4)
    ax.plot(epoch, val_loss, c="tab:orange", label="Validation", lw=4)
    ax.set_xlabel("Iterations", fontsize=18)
    ax.set_ylabel("MAE", fontsize=18)
    ax.tick_params(direction="in")
    ax.legend()

    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def add_feature_inportance_pie_to_plot(ax, importances, feature_names, run_ident, path="results/feature_importance_analysis/"):
    print(run_ident)
    size = 0.3
    # grouped in coordination spheres (up to 3rd) by most metal distant atom (RACs); reminder: {start}-{prop}-{d}-{scope}
    # not complete (may append on demand)
    feature_classes = {"Metal": ["I(M)", "Ox", r"sum($\chi$)", r"min($\chi$)", r"max($\chi$)", "S", "SS", "delE", 'mc-chi-0-all',
                                 'mc-S-0-all', 'mc-Z-0-all', ],
                       "Ligand" : ["CA", "LC", "LD", "#A", "L#A", "max_LBO", "K", "TK"],
                       "Counts" : ["#B", "#C", "#N", "#O", "#F", "#P", "#S", "#Cl", "#Br", "#I",
                                         "T#B", "T#C", "T#N", "T#O", "T#F", "T#P", "T#S", "T#Cl", "T#Br", "T#I"],
                       "1st" : ['D_mc-chi-1-all', 'D_mc-Z-1-all', 'mc-Z-1-all'],
                       "2nd" : ['mc-chi-2-all', 'D_mc-chi-2-all'],
                       "3rd" : ['D_lc-S-3-ax1', 'D_lc-S-3-ax2', 'D_lc-S-3-ax3', 'D_lc-S-3-ax4',
                                'D_lc-T-3-ax1', 'D_lc-T-3-ax2', 'D_lc-T-3-ax3', 'D_lc-T-3-ax4',
                                'lc-Z-2-ax1', 'lc-Z-2-ax2', 'lc-Z-2-ax3', 'lc-Z-2-ax4',
                                'D_lc-S-3-ax1', 'D_lc-S-3-ax2', 'D_lc-S-3-ax3', 'D_lc-S-3-ax4',
                                'mc-Z-3-all'],
                       "Global" : ['f-chi-0-ax1', 'f-chi-0-ax2', 'f-chi-0-ax3', 'f-chi-0-ax4',
                                   'f-Z-1-all', 'f-Z-1-ax1' 'f-Z-1-ax2' 'f-Z-1-ax3' 'f-Z-1-ax4']
                       }
    """
    blues:  ['#f7fbff', '#ecf4fb', '#e1edf8', '#d6e6f4', '#ccdff1', '#bdd7ec', '#abd0e6', '#99c7e0', '#82bbdb', '#6aaed6', '#58a1cf', '#4695c8', '#3787c0', '#2979b9', '#1b69af', '#105ba4', '#084d96', '#083e81', '#08306b']
    greens:  ['#f7fcf5', '#ebf7e7', '#dbf1d6', '#c7e9c0', '#aedea7', '#92d28f', '#73c476', '#52b365', '#37a055', '#228a44', '#0b7734', '#005f26', '#00441b']
    oranges:  ['#fff5eb', '#fee8d2', '#fdd5ad', '#fdb97d', '#fd9c51', '#f87d29', '#e95e0d', '#cd4401', '#a13403', '#7f2704']
    reds:  ['#fff5f0', '#fee3d6', '#fcc4ad', '#fca082', '#fb7c5c', '#f6553c', '#e32f27', '#c2161b', '#9d0d14', '#67000d']
    greys:  ['#ffffff', '#f9f9f9', '#f4f4f4', '#ededed', '#e4e4e4', '#dcdcdc', '#d1d1d1', '#c6c6c6', '#bbbbbb', '#adadad', '#9e9e9e', '#8f8f8f', '#828282', '#757575', '#686868', '#5c5c5c', '#4d4d4d', '#3c3c3c', '#2b2b2b', '#1c1c1c', '#0e0e0e', '#000000']
    """
    inner_chart_colors = {"Global" : {"S" : "#BABABA", "Z" : "#D1D1D1", "chi" : "#DDDDDD", "ox" : "#F4F4F4"},
                          "Counts" : {'#B': '#ffffff', '#C': '#f9f9f9', '#N': '#f4f4f4', '#O': '#ededed', '#F': '#e4e4e4',
                                      '#P': '#dcdcdc', '#S': '#d1d1d1', '#Cl': '#c6c6c6', '#Br': '#bbbbbb', '#I': '#adadad',
                                      'T#B': '#9e9e9e', 'T#C': '#8f8f8f', 'T#N': '#828282', 'T#O': '#757575', 'T#F': '#686868',
                                      'T#P': '#5c5c5c', 'T#S': '#4d4d4d', 'T#Cl': '#3c3c3c', 'T#Br': '#2b2b2b', 'T#I': '#1c1c1c'},
                          "Metal" : {"S" : "#6666FF", "T" : "#9898FF", "Z" : "#B1B2FF", "chi" : "#CCCCFF",
                                     "I(M)": '#f7fbff', "Ox": '#e1edf8', r"sum($\chi$)": '#d6e6f4', r"min($\chi$)": '#ccdff1',
                                     r"max($\chi$)": '#bdd7ec', "SS": '#3787c0', "delE": '#084d96'},
                          "1st" : {"S" : "#FF6666", "T" : "#FF8080", "Z" : "#FF9999", "chi" : "#FFB2B3"},
                          "Ligand" : {"CA" : "#FF6666", "LC" : "#FF8080", "LD" : "#FF9999", "#A" : "#FFB2B3", "L#A": '#fee3d6',
                                      "max_LBO": '#fdb97d', "K": '#e32f27', "TK": '#c2161b'},
                          "2nd" : {"S" : "#65DC8C", "T" : "#80E19F", "Z" : "#99E7B3", "chi" : "#B2EEC6"},
                          "3rd" : {"S" : "#FFC966", "T" : "#FFD280", "Z" : "#FFDC99", "chi" : "#FFDC99"}
                          }
    outer_chart_colors = {"Global" : '#999CA1',
                          "Counts" : '#999CA1',
                          "Metal" : '#4356A3',
                          "1st" : '#F04646',
                          "Ligand" : '#F04646',
                          "2nd" : '#4EBA70',
                          "3rd" : '#FEB137'
                          }

    outer_chart_colors = {'Global' : 0, 'Counts': 0, 'Metal': 0, 'Ligand': 0, '1st' : 0, '2nd' : 0, '3rd' : 0}
    inner_chart_colors = {"Global" : {"S" : 0, "Z" : 0, "chi" : 0, "ox" : 0},
                          "Counts" : {'#B': 0, '#C': 0, '#N': 0, '#O': 0, '#F': 0,
                                      '#P': 0, '#S': 0, '#Cl': 0, '#Br': 0, '#I': 0,
                                      'T#B': 0, 'T#C': 0, 'T#N': 0, 'T#O': 0, 'T#F': 0,
                                      'T#P': 0, 'T#S': 0, 'T#Cl': 0, 'T#Br': 0, 'T#I': 0},
                          "Metal" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0,
                                     "I(M)": 0, "Ox": 0, r"sum($\chi$)": 0, r"min($\chi$)": 0,
                                     r"max($\chi$)": 0, "SS": 0, "delE": 0},
                          "1st" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
                          "Ligand" : {"CA" : 0, "LC" : 0, "LD" : 0, "#A" : 0, "L#A": 0,
                                      "max_LBO": 0, "K": 0, "TK": 0},
                          "2nd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
                          "3rd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0}
                          }

    colormap_selection = ["Greys_r", "Purples_r", "Blues_r", "Greens_r", "Greens_r", "Reds_r", "Oranges_r"]
    for colormap, key in zip(colormap_selection, outer_chart_colors):
        cmap = cm.get_cmap(colormap, len(inner_chart_colors[key].keys()) + 5)

        outer_chart_colors[key] = colors.rgb2hex(cmap(2))
        for i, key2 in enumerate(inner_chart_colors[key]):
            inner_chart_colors[key][key2] = colors.rgb2hex(cmap(i + 3))

    # since features in eaach subclass (meatal, Ligand, Counts) are not equally long
    # and feature labels are of type string, we use an intermediate helper dictionary
    # for the feature grouping
    # vals = {'Metal': {}, 'Ligand': {}, 'Counts': {}, '1st' : {}, '2nd' : {}, '3rd' : {}, 'Global' : {}}
    vals = {"Global" : {"S" : 0, "Z" : 0, "chi" : 0, "ox" : 0},
            "Counts" : {'#B': 0, '#C': 0, '#N': 0, '#O': 0, '#F': 0,
                        '#P': 0, '#S': 0, '#Cl': 0, '#Br': 0, '#I': 0,
                        'T#B': 0, 'T#C': 0, 'T#N': 0, 'T#O': 0, 'T#F': 0,
                        'T#P': 0, 'T#S': 0, 'T#Cl': 0, 'T#Br': 0, 'T#I': 0},
            "Metal" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0,
                       "I(M)": 0, "Ox": 0, r"sum($\chi$)": 0, r"min($\chi$)": 0,
                       r"max($\chi$)": 0, "SS": 0, "delE": 0},
            "1st" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
            "Ligand" : {"CA" : 0, "LC" : 0, "LD" : 0, "#A" : 0, "L#A": 0,
                        "max_LBO": 0, "K": 0, "TK": 0},
            "2nd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0},
            "3rd" : {"S" : 0, "T" : 0, "Z" : 0, "chi" : 0}
            }
    assert len(importances) == len(feature_names)
    print("feature names", feature_names)
    for i, feature_name in enumerate(feature_names):
        for class_i in feature_classes:
            if feature_name in feature_classes[class_i]:
                split = feature_name.split('-')
                feature_name = feature_name if len(split) < 2 else split[1]
                vals[class_i][feature_name] += importances[i]
    # remove zero valued feature names
    vals_cleaned = {'Metal': {}, 'Ligand': {}, 'Counts': {}, '1st' : {}, '2nd' : {}, '3rd' : {}, 'Global' : {}}
    for key in vals:
        for key2 in vals[key]:
            if vals[key][key2] != 0:
                print(f"deleting {key2} from {key}")
                vals_cleaned[key][key2] = vals[key][key2]
        if vals_cleaned[key] == {}:
            del vals_cleaned[key]
    vals = vals_cleaned
    # count number of features that are in each subclass
    outer_values = []
    outer_colors = []
    outer_names = []
    inner_values = []
    inner_colors = []
    inner_names = []
    for i, class_i in enumerate(vals):
        lock = False
        for j, feature_name in enumerate(vals[class_i]):
            if not lock:
                outer_colors += [outer_chart_colors[class_i]]
                outer_names += [class_i]
                outer_values += [0]
                lock = True
            outer_values[-1] += vals[class_i][feature_name]
            inner_values += [vals[class_i][feature_name]]
            inner_colors += [inner_chart_colors[class_i][feature_name]]
            inner_names += [feature_name if feature_name not in ["chi"] else r"$\chi$"]
    print(outer_names)
    print(inner_names)

    size = 0.3
    radius = 1
    outer_pie = ax.pie(outer_values, radius=radius, colors=outer_colors,
                       wedgeprops=dict(width=size, edgecolor='w'), labeldistance=radius - size * 0.5,
                       rotatelabels=True, textprops={'fontsize': 20})

    inner_pie = ax.pie(inner_values, radius=radius - size, colors=inner_colors,
                       labeldistance=0.9, wedgeprops=dict(width=0.4, edgecolor='w'), rotatelabels=True, textprops={'fontsize': 200})

    plt.setp(inner_pie[1], rotation_mode="anchor", ha="left", va="center")
    plt.setp(outer_pie[1], rotation_mode="anchor", ha="center", va="center")
    inner_values_norm = np.array(inner_values) / np.max(inner_values)
    outer_values_norm = np.array(outer_values) / np.max(outer_values)

    for val, label, c, t, w in zip(inner_values_norm, inner_names, inner_colors, inner_pie[1], inner_pie[0]):
        x, y = t.get_position()
        angle = int(np.degrees(np.arctan2(y, x)))
        ha = "right"

        if x < 0:
            angle -= 180
            ha = "left"
        # if val < 0.05:
        #     ang = (w.theta2 - w.theta1) / 2. + w.theta1
        #     y = np.sin(np.deg2rad(ang))
        #     x = np.cos(np.deg2rad(ang))
        #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        #     connectionstyle = f"angle,angleA=0,angleB={ang}"
        #     bbox_props = dict(boxstyle="square,pad=0.3", fc=c, ec=c, lw=0.72)
        #     kw = dict(arrowprops=dict(arrowstyle="-", color=c, linewidth=2),
        #               bbox=bbox_props, zorder=1, va="center")
        #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #     ax.annotate(label, xy=(x, y), xytext=(text_xscale * x, text_yscale * y),
        #                 horizontalalignment=horizontalalignment, **kw)
        # else:
        print("adding label: ", label)
        ax.annotate(label, xy=(x, y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=8)
    for val, c, label, t, w in zip(outer_values_norm, outer_colors, outer_names, outer_pie[1], outer_pie[0]):
        if val < 0.05:
            ang = (w.theta2 - w.theta1) / 2. + w.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            bbox_props = dict(boxstyle="square,pad=0.3", fc=c, ec=c, lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-", color=c, linewidth=2),
                      bbox=bbox_props, zorder=0, va="center")
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(label, xy=(x, y), xytext=(1.2 * np.sign(x), y),
                        horizontalalignment=horizontalalignment, **kw)
        else:
            x, y = t.get_position()
            angle = int(np.degrees(np.arctan2(y, x)) + 90)
            if x < 0:
                angle -= 180
            ha = "center"
            ax.annotate(label, xy=(x, y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=10, c="white")
    ax.set_axis_off()
    plt.savefig(f"{path}/{run_ident}.pdf", dpi=300, bbox_inches='tight')
