#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright 2021, Jonas Oldenstaedt <joldenstaedt@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from sklearn.inspection import permutation_importance
from fcTMCml.constants import feature_target_dir, cache_dir
from fcTMCml.tools import make_dir, load_features, get_pca, get_tsne, get_umap, plot_pca, plot_tsne, plot_umap

from sklearn.ensemble import RandomForestRegressor

from matplotlib import colors, cm

np.random.seed(128)


def run_rf(X: np.array, y: np.array) -> RandomForestRegressor:
    # RFC
    print("\n RFC")
    # kf = KFold(n_splits=10, shuffle=True, random_state=185)
    model = RandomForestRegressor(n_estimators=1000, criterion='squared_error', min_samples_leaf=1, max_leaf_nodes=None, bootstrap=True, oob_score=True,
                                  random_state=128, ccp_alpha=0.0, max_samples=None)
    model.fit(X, y)
    return model


def select_features_permutation_importance(A: np.array, y_truth: np.array, run_ident: str,
                                           cache_dir: str, permutation_importance_subdir: str,
                                           feature_names: np.array, feature_groups: np.array = None,
                                           init_run: bool = True,
                                           maximum_retained_features: int = -1) -> (np.array, np.array, np.array):
    """
    calculate most important features based on random forest permutation importance

    Parameters
    ----------
    A: np.array
        input feature vector
    y_truth: np.array
        input prediction target (ground truth) vector
    run_ident: str
        feature identifier for storing permuation importance results in cache
    cache_dir: str
        directory where to cache the permutation importances
    permutation_importance_subdir: str
        path of permutation importance results in the cache
        (either empty or existend depending on init_run parameter)
    feature_names: np.array
        array with feature names
    feature_groups: np.array (default None)
        if features are supposed to be grouped, this array maps each feature to a groupßß
    init_run: bool
        usually true, if false the function tries to load the permutation importances from the provided cache_dir
    maximum_retained_features: int
        modify threshold of permutation importance to retain at most n features;
        if -1 (default) all features that fall above a certain threshold are kept

    Returns
    -------
    selected_features: np.array
        new feature vector with only the selected features
    """

    if init_run:
        model = run_rf(A, y_truth.flatten())
        result = permutation_importance(model, A, y_truth.flatten(), n_repeats=50, random_state=128)
        np.save(cache_dir + permutation_importance_subdir + run_ident + ".npy", result.importances_mean)

    result_importances_mean = np.load(cache_dir + permutation_importance_subdir + run_ident + ".npy")

    if feature_groups is None:
        result_importances_mean_grouped = result_importances_mean
    else:
        result_importances_mean_grouped = np.bincount(feature_groups, weights=result_importances_mean)

    thres = 0.010
    # print(maximum_retained_features)
    if maximum_retained_features > -1:
        while np.count_nonzero(list((result_importances_mean_grouped / np.max(result_importances_mean_grouped)) > thres)) > maximum_retained_features:
            thres += 0.0005

    mask = np.where((result_importances_mean_grouped / np.max(result_importances_mean_grouped)) > thres)[0]
    mask = mask[np.argsort(result_importances_mean_grouped[mask])]
    print("mask: ", mask)
    selected_features = []
    for curr_group_index in mask:
        print(np.where(feature_groups == curr_group_index))
        print(feature_names[np.where(feature_groups == curr_group_index)[0]])
        selected_features += list(A.T[np.where(feature_groups == curr_group_index)[0]].T)
    quit()
    print(selected_features)
    selected_features = A.T[mask].T
    selected_feature_names = list(feature_names[mask])
    selected_feature_importances = list(result_importances_mean[mask])
    print("retained ", selected_features.shape[1], " features")
    return selected_features, selected_feature_names, selected_feature_importances

##################
# Classification #
##################


# set up folder structure classification
classification_permutation_importances_cache_subdir = "classification_permutation_importances_mean/"

make_dir(cache_dir)
make_dir(cache_dir + classification_permutation_importances_cache_subdir)

classification_in_subdir = "classification_balanced/"

classification_out_subdir = "classification_rff_selection/"

pca_subdir = "pca/"
tsne_subdir = "tsne/"
umap_subdir = "umap/"

make_dir(feature_target_dir + classification_out_subdir)
make_dir(feature_target_dir + classification_out_subdir + pca_subdir)
make_dir(feature_target_dir + classification_out_subdir + tsne_subdir)
make_dir(feature_target_dir + classification_out_subdir + umap_subdir)

# load features
classification_targets, feature_dict, feature_names_dict, feature_groups_dict = load_features(feature_target_dir, classification_in_subdir,
                                                                                              "classification", load_groups=True)

# loop over all feature sets
# 1. pre feature selection PCA
# 2. feature selection
# 3. post feature selection PCA

for run_ident in feature_dict:
    features = feature_dict[run_ident]
    feature_names = feature_names_dict[run_ident]
    feature_groups = feature_groups_dict[run_ident]
    principalComponents, explained_variance = get_pca(features=features)
    tsne_embedding = get_tsne(features=features)
    umap_embedding = get_umap(features=features)
    plot_tsne(tsne_embedding, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
              feature_target_dir + classification_out_subdir + tsne_subdir + run_ident + "_before_RF", legends=["THD", "SQP"])
    plot_umap(umap_embedding, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
              feature_target_dir + classification_out_subdir + umap_subdir + run_ident + "_before_RF", legends=["THD", "SQP"])
    plot_pca(principalComponents, explained_variance, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
             feature_target_dir + classification_out_subdir + pca_subdir + run_ident + "_before_RF", legends=["THD", "SQP"])

    selected_features, selected_feature_names, selected_feature_importances = \
        select_features_permutation_importance(features, classification_targets,
                                               feature_names=feature_names,
                                               feature_groups=feature_groups,
                                               run_ident=run_ident,
                                               cache_dir=cache_dir,
                                               permutation_importance_subdir=classification_permutation_importances_cache_subdir,
                                               init_run=False,
                                               maximum_retained_features=10)
    print(selected_feature_names)

    principalComponents, explained_variance = get_pca(features=selected_features)
    tsne_embedding = get_tsne(features=selected_features)
    umap_embedding = get_umap(features=selected_features)
    plot_tsne(tsne_embedding, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
              feature_target_dir + classification_out_subdir + tsne_subdir + run_ident + "_after_RF", legends=["THD", "SQP"])
    plot_umap(umap_embedding, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
              feature_target_dir + classification_out_subdir + umap_subdir + run_ident + "_after_RF", legends=["THD", "SQP"])
    plot_pca(principalComponents, explained_variance, np.where(classification_targets == 0, "tab:blue", "tab:orange"),
             feature_target_dir + classification_out_subdir + pca_subdir + run_ident + "_after_RF", legends=["THD", "SQP"])

    # TODO(jonas): write generic function for feauture storing
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}.npy", selected_features)
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}_names.npy", selected_feature_names)
    np.save(feature_target_dir + classification_out_subdir + f"{run_ident}_importances.npy", selected_feature_importances)
    np.save(feature_target_dir + classification_out_subdir + "classification_targets.npy", classification_targets)

##############
# Regression #
##############

# set up folder structure regression
regression_permutation_importances_cache_subdir = "regression_permutation_importances_mean/"

make_dir(cache_dir)
make_dir(cache_dir + regression_permutation_importances_cache_subdir)

regression_in_subdir = "regression_raw/"

regression_out_subdir = "regression_rff_selection/"

pca_subdir = "pca/"
tsne_subdir = "tsne/"
umap_subdir = "umap/"


make_dir(feature_target_dir + regression_out_subdir)

make_dir(feature_target_dir + regression_out_subdir)
make_dir(feature_target_dir + regression_out_subdir + pca_subdir)
make_dir(feature_target_dir + regression_out_subdir + tsne_subdir)
make_dir(feature_target_dir + regression_out_subdir + umap_subdir)


# load features
regression_targets, feature_dict, feature_names_dict, feature_groups_dict = load_features(feature_target_dir, regression_in_subdir,
                                                                                          "regression", load_groups=True)

# loop over all feature sets
# 1. pre feature selection PCA
# 2. feature selection
# 3. post feature selection PCA

for run_ident in feature_dict:
    features = feature_dict[run_ident]
    feature_names = feature_names_dict[run_ident]
    feature_groups = feature_groups_dict[run_ident]
    principalComponents, explained_variance = get_pca(features=features)
    tsne_embedding = get_tsne(features=features)
    umap_embedding = get_umap(features=features)

    # define colormap for SSE coloring
    cmap = cm.winter
    norm = colors.Normalize(vmin=np.min(regression_targets), vmax=np.max(regression_targets))
    sm = [cm.ScalarMappable(cmap=cmap, norm=norm), "SSE"]
    print(cmap(norm(regression_targets)))
    plot_tsne(tsne_embedding, cmap(norm(regression_targets)),
              feature_target_dir + regression_out_subdir + tsne_subdir + run_ident + "_before_RF", mapper=sm)
    plot_umap(umap_embedding, cmap(norm(regression_targets)),
              feature_target_dir + regression_out_subdir + umap_subdir + run_ident + "_before_RF", mapper=sm)
    plot_pca(principalComponents, explained_variance, cmap(norm(regression_targets)),
             feature_target_dir + regression_out_subdir + pca_subdir + run_ident + "_before_RF", mapper=sm)

    selected_features, selected_feature_names, selected_feature_importances = \
        select_features_permutation_importance(features, regression_targets, feature_names=feature_names, 
                                               feature_groups=feature_groups, run_ident=run_ident,
                                               cache_dir=cache_dir, permutation_importance_subdir=regression_permutation_importances_cache_subdir,
                                               init_run=False, maximum_retained_features=10)
    print(selected_feature_names)

    principalComponents, explained_variance = get_pca(features=selected_features)
    tsne_embedding = get_tsne(features=selected_features)
    umap_embedding = get_umap(features=selected_features)
    plot_tsne(tsne_embedding, cmap(norm(regression_targets)),
              feature_target_dir + regression_out_subdir + tsne_subdir + run_ident + "_after_RF", mapper=sm)
    plot_umap(umap_embedding, cmap(norm(regression_targets)),
              feature_target_dir + regression_out_subdir + umap_subdir + run_ident + "_after_RF", mapper=sm)
    plot_pca(principalComponents, explained_variance, cmap(norm(regression_targets)),
             feature_target_dir + regression_out_subdir + pca_subdir + run_ident + "_after_RF", mapper=sm)

    # TODO(jonas): write generic function for feauture storing
    np.save(feature_target_dir + regression_out_subdir + f"{run_ident}.npy", selected_features)
    np.save(feature_target_dir + regression_out_subdir + f"{run_ident}_names.npy", selected_feature_names)
    np.save(feature_target_dir + regression_out_subdir + f"{run_ident}_importances.npy", selected_feature_importances)
    np.save(feature_target_dir + regression_out_subdir + "regression_targets.npy", regression_targets)
