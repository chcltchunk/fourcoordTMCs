import numpy as np
import pickle as pkl
import re

from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, Trials
from functools import partial


from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features


# geometry 0 : "tedrahedral", 1 : "square planar", 2 : "seesaw"
geometry = []
feature_list = []
path_control = []  # used to check whether the split is reasonable

strict_cutoff = 0
catom_list = None


def k_folds(clf, X, y, return_clf=False, rns=25):
    kf = KFold(n_splits=10, shuffle=True, random_state=128s)
    kf.get_n_splits(X)
    accuracy_score = []
    ppvs = []
    sensitivities = []
    f_scores = []
    for train_index, test_index in kf.split(X):
        # print("test set contains:\n ", path_control[test_index])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train, y_train)
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
    # RidgeClassifier
    print("\n RidgeClassifier")
    alpha_n = np.logspace(0.001, 1, 20)
    tol = [1e-3, 1e-4, 1e-5]
    grid = dict(alpha=alpha_n, tol=tol)
    kf = KFold(n_splits=10, shuffle=True, random_state=1285)
    clf = RidgeClassifier(random_state=1288)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X,y)

    # print("avg score: ", k_folds(RidgeClassifier(alpha=0.6), curr_X, y))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 
    

def grid_search_svc(X, y):
    # SVC
    print("\n SVC")
    kernel = ['linear']  # , 'rbf' , 'poly']#, 'rbf', 'sigmoid']
    C = [1000]  # , 100] #  [1010, 1000, 990]#, 500, 100, 50]#, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    degree = [3]  # [2, 3, 4, 5]
    coef0 = [2.8]  # np.linspace(2.6, 2.9, 20)
    class_weight = [None]  # , "balanced"]
    grid = dict(kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight=class_weight)  # , coef0=coef0)
    clf = SVC(random_state=1288)
    kf = KFold(n_splits=10, shuffle=True, random_state=1285)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X,y)
    # print("Ridge regression; alpha: ", alpha, " avg score: ", k_folds(clf))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_


def grid_search_rfc(X, y):
    # RFC
    print("\n RFC")
    clf = RandomForestClassifier(random_state=1288)
    n_estimators = [10, 100, 1000]
    max_features = ["log2", 'sqrt', 'log2']
    criterion = ["entropy", "gini"]
    min_samples_split = [.00001, .0001, 0.001, 0.01, 0.1, 0.2]
    # min_samples_leaf = [.00001, .0001, .001, .001, .01]
    grid = dict(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, criterion=criterion)
    kf = KFold(n_splits=10, shuffle=True, random_state=1285)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f & (%f) & %r \\\\" % (mean, stdev, param))
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
    return 1 - acc


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
    # print("---hyperopt---")
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
    return 1 - acc


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
    # print("---hyperopt---")
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
                rstate=np.random.default_rng(128)
                )

    print("best_perams: ", best)
    return best


def run_rf(X: np.array, y: np.array) -> RandomForestRegressor:
    # RFC
    print("\n RFC")
    # kf = KFold(n_splits=10, shuffle=True, random_state=1285)
    model = RandomForestRegressor(n_estimators=1000, criterion='squared_error', min_samples_leaf=1, max_leaf_nodes=None,
                                  bootstrap=True, oob_score=True, random_state=1288, ccp_alpha=0.0, max_samples=None)
    model.fit(X, y)
    return model


def run_krr(X, y):
    # KRR
    print("\n KRR")
    alpha_n = np.logspace(-12, 0, 50)  # default for this project is 100 gridpoints
    # alpha_n = np.logspace(-12, -5, 20) # default for this project is 100 gridpoints
    gamma_n = np.logspace(-12, 1, 50)  # default for this project is 100 gridpoints
    # gamma_n = np.logspace(-12, -5, 20) # default for this project is 100 gridpoints
    grid = dict(alpha=alpha_n, gamma=gamma_n)
    kf = KFold(n_splits=10, shuffle=True, random_state=1285)
    clf = KernelRidge(kernel='rbf')
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=kf, scoring='neg_mean_squared_error')
    # grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=kf, scoring='neg_mean_absolute_error')
    print(X.shape, y.shape)
    grid_result = grid_search.fit(X,y)

    # print("avg score: ", k_folds(RidgeClassifier(alpha=0.6), curr_X, y))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f & (%f) & %r \\\\" % (mean, stdev, param))
    return grid_result.best_params_ 
    # optimize shuffling
    # for i in np.arange(1, 200):
    #    print(i, "Ridge regression; alpha: ", alpha, " avg score: ", k_folds(clf, rns=i))


classification_in_subdir = "classification_rff_selection/"


classification_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, classification_in_subdir, "classification")

acc_dict = {}
for run_ident in feature_dict:
    X = feature_dict[run_ident]
    y_truth = classification_targets[run_ident]
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    # print(y_test)
    hyperparams = rc_optimization(X_train, X_test, y_train, y_test)
    print(hyperparams)

    clf = RidgeClassifier(**hyperparams)
    avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, X, y_truth, True)
    coeffs = clf.coef_

    print(run_ident + " RR (TPE): ", avg_score, ppv, sensitivity, f_score, coeffs)

    acc_dict[run_ident + " RR (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score":f_score, "coeffs":coeffs}

    # print(y_test)
    hyperparams = rfc_optimization(X_train, X_test, y_train, y_test)
    print(hyperparams)

    clf = RandomForestClassifier(**hyperparams)
    avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, X, y_truth, True)
    coeffs = clf.feature_importances_

    print(run_ident + " RFC (TPE): ", avg_score, ppv, sensitivity, f_score, coeffs)

    acc_dict[run_ident + " RFC (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score": f_score, "coeffs": coeffs}
