from __future__ import print_function
import os
import numpy as np


from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split

from hyperopt import hp, tpe, fmin, Trials, space_eval
from functools import partial


from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features


def k_folds(clf, X, y, return_clf=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=128)
    kf.get_n_splits(X)
    mse_n = []
    mse_n_train = []
    mae_n = []
    mae_n_train = []
    r2_n = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test).reshape(-1, 1)
        pred_train = clf.predict(X_train).reshape(-1, 1)
        mse = metrics.mean_squared_error(y_test, pred)
        mse_train = metrics.mean_squared_error(y_train, pred_train)
        mae = metrics.mean_absolute_error(y_test, pred)
        mae_train = metrics.mean_absolute_error(y_train, pred_train)
        r2 = metrics.r2_score(y_test, pred)
        mse_n += [mse]
        mse_n_train += [mse_train]
        mae_n += [mae]
        mae_n_train += [mae_train]
        r2_n += [r2]
    if return_clf:
        return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train), np.average(mae_n_train), clf
    return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train), np.average(mae_n_train)


def train_krr_hyperopt(hyperparams, X_train, X_val, y_train, y_val, return_model=False) -> float:
    '''
    Train a KernelRidge model with given hyperparameters.

    Parameters
    ----------
    hyperparams: dict
        hyperparameters for KRR.
    X_train: np.array
        training data inputs
    y_train: np.array
        training data targets
    X_val: np.array
        val data inputs
    y_val: np.array
        val data targets
    return_model: bool
        if True, return model instead of MAE

    Returns
    -------
    mae: float
        mean absolute error for the RF model (return_model=False)
    krr: scikit-learn KernelRidge object
        trained KRR model (return_model=True)
    '''
    krr = KernelRidge(alpha=hyperparams["alpha"], gamma=hyperparams["gamma"], kernel=hyperparams['kernel'])
    # krr = KernelRidge(alpha=hyperparams["alpha"], gamma=hyperparams["gamma"], kernel="rbf")
    krr.fit(X_train, y_train)
    if return_model:
        return krr
    y_pred = krr.predict(X_val)
    mse = metrics.mean_squared_error(y_val, y_pred)
    return mse


def krr_optimization(X_train: np.array, X_val: np.array, y_train: np.array, y_val: np.array, space: dict) -> dict:
    '''
    RidgeClassifier hyperparameters optimization with hyperopt.

    Parameters
    ----------
    X_train: np.array
        training data inputs
    y_train: np.array
        training data targets
    X_val: np.array
        val data inputs
    y_val: np.array
        val data targets

    Returns
    -------
    best: dict
        best hyperparameters
    '''
    # print("---hyperopt---")

    objective_func = partial(train_krr_hyperopt,
                             X_train=X_train,
                             X_val=X_val,
                             y_train=y_train,
                             y_val=y_val)
    trials = Trials()
    best = fmin(objective_func,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=100,
                rstate=np.random.default_rng(128)
                )

    print("best_perams: ", best)
    return best


def grid_search_krr(X, y):
    # KRR
    print("\n KRR")
    alpha_n = np.logspace(-12, 0, 50)
    gamma_n = np.logspace(-12, 1, 50)
    grid = dict(alpha=alpha_n, gamma=gamma_n)
    kf = KFold(n_splits=5, shuffle=True, random_state=185)
    clf = KernelRidge(kernel='rbf')
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=kf, scoring='neg_mean_squared_error')
    print(X.shape, y.shape)
    grid_result = grid_search.fit(X, y)
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_


# set random seed
random_seed = 423890532
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)


# import data
regression_in_subdir = "regression_rff_selection/"

regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")


#######
# KRR #
#######
space = {"alpha": hp.quniform("alpha", 1.0, 10.0, 0.01),
        #  "alpha": hp.loguniform("alpha", -8, 1),
         "gamma": hp.loguniform("gamma", -12, 1),
         "kernel": hp.choice("kernel", ["rbf", "laplacian"])
         }
acc_dict = {}
for run_ident in feature_dict:
    X = feature_dict[run_ident]
    print("std: ", np.std(regression_targets))
    y_truth = regression_targets.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    best_hyperparams = krr_optimization(X_train, X_test, y_train, y_test, space)
    hyperparams = space_eval(space, best_hyperparams)
    print(hyperparams)
    clf = KernelRidge(**hyperparams)
    mse, mae, r2, mse_train, mae_train, clf = k_folds(clf, X, y_truth.ravel(), True)

    print(run_ident + " KRR (TPE): ", np.round(mae, 3), np.round(mae_train, 3), np.round(mse, 2), np.round(mse_train, 2), np.round(r2, 2))

    acc_dict[run_ident + " KRR (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "mae_train": mae_train, "r2": r2}


print("Feature Set, ML Model, MAE, MAE_train, R2, MSE (K-Fold), MSE_train")
for key in acc_dict:
    print(", ".join(key.split(" ")[:-2]), ' ,',
          np.round(acc_dict[key]['mae'], 3), ' ,',
          np.round(acc_dict[key]['mae_train'], 3), ' ,',
          np.round(acc_dict[key]['r2'], 3), ' ,',
          np.round(acc_dict[key]['mse'], 3), ' ,',
          np.round(acc_dict[key]['mse_train'], 3), ' ,')

print("")
for key in acc_dict:
    print(" ".join(key.split("_")), ' ,',
          np.round(acc_dict[key]['mae_train'], 3), ' &',
          np.round(np.sqrt(acc_dict[key]['mse_train']), 3), ' &',
          np.round(acc_dict[key]['mae'], 3), ' &',
          np.round(np.sqrt(acc_dict[key]['mse']), 3), ' &',
          np.round(acc_dict[key]['r2'], 3)
          )
