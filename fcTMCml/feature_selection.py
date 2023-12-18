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
import pickle as pkl
import re
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as mpatches

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from molSimplify.Classes.mol3D import mol3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split 




from hyperopt import hp, tpe, fmin, Trials
from functools import partial
from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import make_dir

classification_in_subdir = "classification_balanced/"
regression_in_subdir = "regression_raw/"

classification_out_subdir = "classification_rff_selection/"
regression_out_subdir = "regression_rff_selection/"

make_dir(feature_target_dir + classification_out_subdir)
make_dir(feature_target_dir + regression_out_subdir)

classification_targets = np.load(feature_target_dir + classification_in_subdir + "classification_targets.npy")
regression_targets = np.load(feature_target_dir + regression_in_subdir + "regression_targets.npy")

# geometry 0 : "tedrahedral", 1 : "square planar", 2 : "seesaw"
geometry = []
feature_list = []
path_control = [] # used to check whether the split is reasonable

strict_cutoff = 0
catom_list = None


def plot_pca(principalComponents, explained_variance, color_list, filename, color_dic=None, mapper=None, legends=None, title=None):
    fig, ax = plt.subplots(figsize=(8,8))

    g = ax.scatter(principalComponents.T[0], principalComponents.T[1], s=25, c=color_list)
    ax.set_xlabel("PC 1 ({})".format(np.round(explained_variance[0],2)), fontsize=20)
    ax.set_ylabel("PC 2 ({})".format(np.round(explained_variance[1],2)), fontsize=20)
    ax.tick_params(which="major", direction="in")

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if color_dic is not None: 
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dic.values()]
        plt.legend(markers, color_dic.keys(), numpoints=1, prop={'size': 15})
    elif mapper is not None:
        cbar = plt.colorbar(mapper[0], ticks=np.arange(-70, 11, 10)) 
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(mapper[1], rotation=90, fontsize=15)
    if legends is not None:
        #TODO
        handles, labels = ax.get_legend_handles_labels()
        patch1 = mpatches.Patch(color="tab:blue", label="THD")
        patch2 = mpatches.Patch(color="tab:orange", label="SQP")
        handles.extend([patch1, patch2])
        ax.legend(handles=handles, fontsize=15)
    if title is not None:
        plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    #plt.show()


def k_folds(clf, X, y, return_clf=False, rns=25):
    kf = KFold(n_splits=10, shuffle=True, random_state=rns)
    kf.get_n_splits(X)
    accuracy_score = []
    ppvs = []
    sensitivities = []
    f_scores = []
    for train_index, test_index in kf.split(X):
        # print("test set contains:\n ", path_control[test_index])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(X_train, y_train)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, pred)
        ppv = metrics.precision_score(y_test, pred)
        sensitivity = metrics.recall_score(y_test, pred)
        f_score = metrics.f1_score(y_test, pred)
        accuracy_score += [score]
        ppvs += [ppv]
        sensitivities += [sensitivity]
        f_scores += [f_score]
    print("accuracy:   %0.3f" % score)
    print(accuracy_score)
    if return_clf:
        return np.average(accuracy_score), np.average(ppvs), np.average(sensitivities), np.average(f_scores), clf
    return np.average(accuracy_score), np.average(ppvs), np.average(sensitivities), np.average(f_scores)
        

def grid_search_ridge_clf(X, y):
    ##### RidgeClassifier
    print("\n RidgeClassifier")
    alpha_n = np.logspace(0.001, 1, 20)
    tol = [1e-3, 1e-4, 1e-5]
    grid = dict(alpha=alpha_n, tol=tol)
    kf = KFold(n_splits=10, shuffle=True, random_state=185)
    clf = RidgeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X,y)
    
    #print("avg score: ", k_folds(RidgeClassifier(alpha=0.6), curr_X, y))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 
    # optimize shuffling
    # for i in np.arange(1, 200):
    #    print(i, "Ridge regression; alpha: ", alpha, " avg score: ", k_folds(clf, rns=i))


def grid_search_svc(X, y):
    ##### SVC
    print("\n SVC")
    kernel = ['linear']#, 'rbf' , 'poly']#, 'rbf', 'sigmoid']
    C = [1000]#, 100] #  [1010, 1000, 990]#, 500, 100, 50]#, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    degree = [3]#[2, 3, 4, 5]
    coef0 = [2.8]#np.linspace(2.6, 2.9, 20)
    class_weight=[None]#, "balanced"]
    grid = dict(kernel=kernel,C=C,gamma=gamma,degree=degree, class_weight=class_weight)#, coef0=coef0)
    clf = SVC()
    kf = KFold(n_splits=10, shuffle=True, random_state=185)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X,y)
    # print("Ridge regression; alpha: ", alpha, " avg score: ", k_folds(clf))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 



def grid_search_rfc(X, y):
    ##### RFC
    print("\n RFC")
    clf = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ["log2", 'sqrt', 'log2']
    criterion = ["entropy", "gini"]
    min_samples_split = [.00001, .0001, 0.001, 0.01, 0.1, 0.2]
    min_samples_leaf = [.00001, .0001, .001, .001, .01]
    grid = dict(n_estimators=n_estimators,max_features=max_features, min_samples_split=min_samples_split, criterion=criterion)
    kf = KFold(n_splits=10, shuffle=True, random_state=185)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X,y)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 

def train_rfc_hyperopt(hyperparams, X_train, X_val, y_train, y_val, return_model=False):
    '''
    Train a RFC model at given hyperparameters.

    inputs
    hyperparams: dict, hyperparameters for KRR.
        X_train: np.array, training data inputs
        y_train: np.array, training data targets
        X_val: np.array, val data inputs
        y_val: np.array, val data targets
        return_model: boolean, if True, return model instead of MAE

    outputs:
        mae: float, mean absolute error for the RF model (return_model=False)
        rfc: scikit-learn KernelRidge object, trained KRR model (return_model=True)
    '''
    rfc = RandomForestClassifier(n_estimators=hyperparams["n_estimators"],max_features=hyperparams["max_features"], min_samples_split=hyperparams["min_samples_split"], criterion=hyperparams["criterion"], min_samples_leaf=hyperparams["min_samples_leaf"])
    rfc.fit(X_train, y_train)
    if return_model:
        return rfc
    y_pred = rfc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    return 1-acc

def rfc_optimization(X_train, X_val, y_train, y_val):
    '''
    KRR hyperparameters optimization with hyperopt.

    inputs:
        X_train: np.array, training data inputs
        y_train: np.array, training data targets
        X_val: np.array, val data inputs
        y_val: np.array, val data targets
        y_scaler: scikit-learn standard scaler object, for targets

    outputs:
        best: dict, best hyperparameters.
    '''
    #print("---hyperopt---")
    max_features = ["log2", 'sqrt']
    criterion = ["entropy", "gini"]
    space = {"n_estimators": hp.randint("n_estimators", 10, 100),
            "max_features": hp.choice("max_features", max_features),
            "criterion": hp.choice("criterion", criterion),
            "min_samples_split": hp.loguniform("min_samples_split", np.log(1e-7), np.log(2e-1)),
            "min_samples_leaf": hp.loguniform("min_samples_leaf", np.log(1e-7), np.log(1e-1)),
             }
    objective_func = partial(train_rfc_hyperopt,
                             X_train=X_train,
                             X_val=X_val,
                             y_train=y_train,
                             y_val=y_val)
    trials = Trials()
    best = fmin(objective_func,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=200,
                rstate=np.random.default_rng(0)
                )
    best.update({"max_features": max_features[best['max_features']],
            "criterion": criterion[best['criterion']]})
    print("best_perams: ", best)
    return best

def train_rc_hyperopt(hyperparams, X_train, X_val, y_train, y_val, return_model=False):
    '''
    Train a KRR model at given hyperparameters.

    inputs
    hyperparams: dict, hyperparameters for KRR.
        X_train: np.array, training data inputs
        y_train: np.array, training data targets
        X_val: np.array, val data inputs
        y_val: np.array, val data targets
        return_model: boolean, if True, return model instead of MAE

    outputs:
        mae: float, mean absolute error for the RF model (return_model=False)
        krr: scikit-learn KernelRidge object, trained KRR model (return_model=True)
    '''
    krc = RidgeClassifier(alpha=hyperparams["alpha"], tol=hyperparams["tol"])
    krc.fit(X_train, y_train)
    if return_model:
        return krc
    y_pred = krc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    return 1-acc

def rc_optimization(X_train, X_val, y_train, y_val):
    '''
    KRR hyperparameters optimization with hyperopt.

    inputs:
        X_train: np.array, training data inputs
        y_train: np.array, training data targets
        X_val: np.array, val data inputs
        y_val: np.array, val data targets
        y_scaler: scikit-learn standard scaler object, for targets

    outputs:
        best: dict, best hyperparameters.
    '''
    #print("---hyperopt---")
    space = {"alpha": hp.loguniform("alpha", np.log(1e-8), 1),
             "tol": hp.loguniform("tol", np.log(1e-12), np.log(1)),
             # "ls": hp.loguniform("ls", np.log(1e-6), np.log(1e6)),
             }
    objective_func = partial(train_rc_hyperopt,
                             X_train=X_train,
                             X_val=X_val,
                             y_train=y_train,
                             y_val=y_val)
    trials = Trials()
    best = fmin(objective_func,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=200,
                    rstate=np.random.default_rng(0)
                )

    print("best_perams: ", best)
    return best


def add_fi_to_plot(ax, coeffs, feature_set_mask, q, avg_score):
    #TODO: improve pie charts : https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html#sphx-glr-gallery-pie-and-polar-charts-nested-pie-py
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


    cmap = cm.Blues(np.linspace(0,1,3*metal_feat_length+1))
    cmap = colors.ListedColormap(cmap[metal_feat_length-1:2*metal_feat_length,:-1])
    inner_colors = cmap.colors[:-1] 
    colors_outer = [cmap.colors[-1]]

    cmap = cm.Greens(np.linspace(0,1,3*ligand_feat_length+1))
    cmap = colors.ListedColormap(cmap[ligand_feat_length-1:2*ligand_feat_length,:-1])
    inner_colors = np.vstack((inner_colors, cmap.colors[:-1]))
    colors_outer += [cmap.colors[-1]]

    if charge_length > 0:
        cmap = cm.Oranges(np.linspace(0,1,3*charge_length+1))
        cmap = colors.ListedColormap(cmap[charge_length-1:2*charge_length,:-1])
        inner_colors = np.vstack((inner_colors, cmap.colors[:-1]))
        colors_outer += [cmap.colors[-1]]
    colors_outer = np.array(colors_outer)
    print(colors_outer)
     
    bigger = ax.pie(outer_sizes, labels=outer_labels, colors=colors_outer,
                     startangle=90, frame=True, labeldistance=0.8, rotatelabels=True)
    smaller = ax.pie(coeffs_abs, labels=feature_names[feature_set_mask],
                      colors=inner_colors, radius=0.7,
                      startangle=90, labeldistance=0.5, rotatelabels=True)
    ax.set_title("set #{}; score: {}".format(q, np.round(avg_score, 2)), s=12)

def run_rf(X: np.array, y: np.array) -> RandomForestRegressor:
    ##### RFC
    print("\n RFC")
    #kf = KFold(n_splits=10, shuffle=True, random_state=185)
    model = RandomForestRegressor(n_estimators=1000, criterion='squared_error', min_samples_leaf=1, max_leaf_nodes=None, bootstrap=True, oob_score=True, random_state=None, ccp_alpha=0.0, max_samples=None)
    model.fit(X,y)
    return model 

def run_krr(X, y):
    ##### KRR
    print("\n KRR")
    alpha_n = np.logspace(-12, 0, 50) # default for this project is 100 gridpoints
    #alpha_n = np.logspace(-12, -5, 20) # default for this project is 100 gridpoints
    gamma_n = np.logspace(-12, 1, 50) # default for this project is 100 gridpoints
    #gamma_n = np.logspace(-12, -5, 20) # default for this project is 100 gridpoints
    grid = dict(alpha=alpha_n, gamma=gamma_n)
    kf = KFold(n_splits=10, shuffle=True, random_state=185)
    clf = KernelRidge(kernel='rbf')
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=kf, scoring='neg_mean_squared_error')
    #grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=kf, scoring='neg_mean_absolute_error')
    print(X.shape, y.shape)
    grid_result = grid_search.fit(X,y)
    
    #print("avg score: ", k_folds(RidgeClassifier(alpha=0.6), curr_X, y))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 
    # optimize shuffling
    # for i in np.arange(1, 200):
    #    print(i, "Ridge regression; alpha: ", alpha, " avg score: ", k_folds(clf, rns=i))





targets = np.load(feature_target_dir + "classifier_targets.npy")

mcdl53_classifier_features = np.load(feature_target_dir + "MCDL53_classifier.npy")
mcdl53_classifier_feature_names = np.load(feature_target_dir + "MCDL53_classifier_names.npy")

mcdl53_cff_classifier_features = np.load(feature_target_dir + "MCDL53_cff_classifier.npy")
mcdl53_cff_classifier_feature_names = np.load(feature_target_dir + "MCDL53_cff_classifier_names.npy")

rac300_classifier_features = np.load(feature_target_dir + "RAC_classifier.npy")
rac300_classifier_feature_names = np.load(feature_target_dir + "RAC_classifier_names.npy")

rac300_cff_classifier_features = np.load(feature_target_dir + "RAC_cff_classifier.npy")
rac300_cff_classifier_feature_names = np.load(feature_target_dir + "RAC_cff_classifier_names.npy")