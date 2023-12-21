import importlib.util
import os
import numpy as np


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import cm, colors
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
    targets = np.load(feature_target_dir + sub_dir + f"{type}_targets.npy")
    
    mcdl53_features = np.load(feature_target_dir + sub_dir + f"MCDL53_{type}.npy")
    mcdl53_feature_names = np.load(feature_target_dir + sub_dir + f"MCDL53_{type}_names.npy")

    mcdl53_cff_features = np.load(feature_target_dir + sub_dir + f"MCDL53_cff_{type}.npy")
    mcdl53_cff_feature_names = np.load(feature_target_dir + sub_dir + f"MCDL53_cff_{type}_names.npy")

    rac300_features = np.load(feature_target_dir + sub_dir + f"RAC_{type}.npy")
    rac300_feature_names = np.load(feature_target_dir + sub_dir + f"RAC_{type}_names.npy")

    rac300_cff_features = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}.npy")
    rac300_cff_feature_names = np.load(feature_target_dir + sub_dir + f"RAC_cff_{type}_names.npy")
    return targets, {f"MCDL53_{type}" : mcdl53_features,
                     f"MCDL53_cff_{type}" : mcdl53_cff_features,
                     f"RAC_{type}" : rac300_features,
                     f"RAC_cff_{type}" : rac300_cff_features
                     }, {f"MCDL53_{type}" : mcdl53_feature_names,
                         f"MCDL53_cff_{type}" : mcdl53_cff_feature_names,
                         f"RAC_{type}" : rac300_feature_names,
                         f"RAC_cff_{type}" : rac300_cff_feature_names
                         }


def get_pca(features: np.array) -> (list, list):
    pca = PCA()
    principalComponents = pca.fit_transform(features)
    explained_variances = pca.explained_variance_ratio_
    return (principalComponents, explained_variances)


def get_tsne(features: np.array) -> list:
    tsne = TSNE()
    tsne_vector = tsne.fit_transform(features)
    return tsne_vector


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


# TODO(jonas): cleanup
def add_fi_to_plot(ax, coeffs, feature_set_mask, q, avg_score):
    # TODO: improve pie charts : https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html#sphx-glr-gallery-pie-and-polar-charts-nested-pie-py
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    size = 0.3
    vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
    # Normalize vals to 2 pi
    valsnorm = vals/np.sum(vals)*2*np.pi
    # Obtain the ordinates of the bar edges
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap([1, 2, 5, 6, 9, 10])

    ax.bar(x=valsleft[:, 0],
       width=valsnorm.sum(axis=1), bottom=1-size, height=size,
       color=outer_colors, edgecolor='w', linewidth=1, align="edge")

    ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=1-2*size, height=size,
       color=inner_colors, edgecolor='w', linewidth=1, align="edge")

    ax.set(title="Pie plot with `ax.bar` and polar coordinates")
    ax.set_axis_off()
    plt.show()

    """
    coeffs_abs = np.abs(coeffs)
    metal_feat_length = np.sum(feature_set_mask[:7])
    charge_length = np.sum(feature_set_mask[-2:])
    ligand_feat_length = len(coeffs_abs) - metal_feat_length - charge_length
    if charge_length > 0:
        outer_sizes = [np.sum(coeffs_abs[:metal_feat_length]), np.sum(coeffs_abs[metal_feat_length:-charge_length]), np.sum(coeffs_abs[-charge_length:])]
        outer_labels = ['metal', 'ligand', 'mulliken charges']
    else:
        outer_sizes = [np.sum(coeffs_abs[:metal_feat_length]), np.sum(coeffs_abs[metal_feat_length:])]
        outer_labels = ['metal', 'ligand'] 

    cmap = cm.Blues(np.linspace(0, 1, 3 * metal_feat_length + 1))
    cmap = colors.ListedColormap(cmap[metal_feat_length - 1:2 * metal_feat_length, :-1])
    inner_colors = cmap.colors[:-1] 
    colors_outer = [cmap.colors[-1]]

    cmap = cm.Greens(np.linspace(0, 1, 3 * ligand_feat_length + 1))
    cmap = colors.ListedColormap(cmap[ligand_feat_length - 1:2 * ligand_feat_length, :-1])
    inner_colors = np.vstack((inner_colors, cmap.colors[:-1]))
    colors_outer += [cmap.colors[-1]]

    if charge_length > 0:
        cmap = cm.Oranges(np.linspace(0, 1, 3 * charge_length + 1))
        cmap = colors.ListedColormap(cmap[charge_length - 1:2 * charge_length, :-1])
        inner_colors = np.vstack((inner_colors, cmap.colors[:-1]))
        colors_outer += [cmap.colors[-1]]
    colors_outer = np.array(colors_outer)
    print(colors_outer)

    # bigger = ax.pie(outer_sizes, labels=outer_labels, colors=colors_outer,
    #                  startangle=90, frame=True, labeldistance=0.8, rotatelabels=True)
    # smaller = ax.pie(coeffs_abs, labels=feature_names[feature_set_mask],
    #                   colors=inner_colors, radius=0.7,
    #                   startangle=90, labeldistance=0.5, rotatelabels=True)
    ax.set_title("set #{}; score: {}".format(q, np.round(avg_score, 2)), s=12)
