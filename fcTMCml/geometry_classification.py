import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from hyperopt import hp, tpe, fmin, Trials
from functools import partial


from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features


def k_folds(clf: ClassifierMixin,
            X: np.array, y: np.array,
            return_clf: bool = False):
    """
    K-Fold cross validation for a Classifier Model

    Parameters:
    -----------
    clf: ClassifierMixin
        the classifier model to run
    X: np.array
        feature vector
    y: np.array
        one-hot encoded target vector
    return_clf: bool
        if True the classifier will be returnee along with k-fold scores

    Returns:
    --------
    accuracy: float
        average of model accuracy over k-folds
    ppv: float
    sensitivities: float
    f_scores: float
    clf: ClassifierMixin
        only if return_clf is set to true
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=128)
    kf.get_n_splits(X)
    accuracy_score = []
    ppvs = []
    sensitivities = []
    f_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clone(clf)
        # verify that the splits are equally distributed among the classes
        # print(np.count_nonzero(y_train), " / ", y_train.shape[0], " square_planar / total datapoints in training set")
        # print(np.count_nonzero(y_test), " / ", y_test.shape[0], " square_planar / total datapoints in testing set")
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
    if return_clf:
        return np.average(accuracy_score), np.average(ppvs), np.average(sensitivities), np.average(f_scores), clf
    return np.average(accuracy_score), np.average(ppvs), np.average(sensitivities), np.average(f_scores)


def grid_search_rfc(X: np.array, y: np.array):
    """
    run grid search for the RandomForestClassifier

    Parameters:
    -----------
    X: np.array
        feature vector
    y: np.array
        one-hot encoded target vector

    Returns:
    --------
    best_hyperparams: dict
        best hyperparameters

    """
    # RFC (takes care of K-Fold internally)
    print("\n RFC")
    clf = RandomForestClassifier(random_state=128)
    n_estimators = [10, 100, 1000]
    max_features = ["log2", 'sqrt', 'log2']
    criterion = ["entropy", "gini"]
    min_samples_split = [.00001, .0001, 0.001, 0.01, 0.1, 0.2]
    grid = dict(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, criterion=criterion)
    kf = KFold(n_splits=10, shuffle=True, random_state=128)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=kf, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X, y)
    return grid_result.best_params_


def train_rfc_hyperopt(hyperparams: dict,
                       X_train: np.array, X_val: np.array,
                       y_train: np.array, y_val: np.array,
                       return_model: bool = True):
    '''
    Train a RandomForestClassifiert model at given hyperparameters.

    Parameters:
    -----------
    hyperparams: dict
        hyperparameters for RFC
    X_train: np.array
        training data inputs
    X_val: np.array
        validation data inputs
    y_train: np.array
        training data targets
    y_val: np.array
        validation data targets
    return_model: bool (default True)
        if True, return model instead of 1 - accuracy

    Returns:
    --------
    1 - accuracy: float
        1 - accuracy to convert this into a minimization problem
    '''
    rfc = RandomForestClassifier(n_estimators=hyperparams["n_estimators"], max_features=hyperparams["max_features"],
                                 min_samples_split=hyperparams["min_samples_split"], criterion=hyperparams["criterion"],
                                 min_samples_leaf=hyperparams["min_samples_leaf"], random_state=128)
    rfc.fit(X_train, y_train)
    if return_model:
        return rfc
    y_pred = rfc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    return 1 - acc


def rfc_optimization(X_train: np.array, X_val: np.array,
                     y_train: np.arary, y_val: np.array):
    '''
    RandomForestClassifier hyperparameters optimization with hyperopt.

    Parameters:
    -----------
    X_train: np.array
        raining data inputs
    y_train: np.array
        training data targets
    X_val: np.array
        validation data inputs
    y_val: np.array
        validation data targets

    Returns:
    --------
    best_hyperparams: dict
        best hyperparameters
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


def train_rc_hyperopt(hyperparams: dict,
                      X_train: np.array, X_val: np.array,
                      y_train: np.array, y_val: np.array,
                      return_model: bool = True) -> float:
    '''
    Train a RidgeClassifier model with given hyperparameters.

    Parameters:
    -----------
    hyperparams: dict
        hyperparameters for RFC
    X_train: np.array
        training data inputs
    X_val: np.array
        validation data inputs
    y_train: np.array
        training data targets
    y_val: np.array
        validation data targets
    return_model: bool (default True)
        if True, return model instead of 1 - accuracy

    Returns:
    --------
    1 - accuracy: float
        1 - accuracy to convert this into a minimization problem
    '''
    rc = RidgeClassifier(alpha=hyperparams["alpha"], tol=hyperparams["tol"], random_state=128)
    rc.fit(X_train, y_train)
    if return_model:
        return rc
    y_pred = rc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    return 1 - acc


def rc_optimization(X_train: np.array, X_val: np.array,
                    y_train: np.array, y_val: np.array) -> dict:
    '''
    RidgeClassifier hyperparameters optimization with hyperopt.

    Parameters:
    -----------
    X_train: np.array
        raining data inputs
    y_train: np.array
        training data targets
    X_val: np.array
        validation data inputs
    y_val: np.array
        validation data targets

    Returns:
    --------
    best_hyperparams: dict
        best hyperparameters
    '''
    # print("---hyperopt---")
    space = {"alpha": hp.loguniform("alpha", np.log(1e-8), 1),
             "tol": hp.loguniform("tol", np.log(1e-12), np.log(1)),
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


classification_in_subdir = "classification_rff_selection/"


classification_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, classification_in_subdir, "classification")

acc_dict = {}
for run_ident in feature_dict:
    X = feature_dict[run_ident]
    y_truth = classification_targets
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)

    hyperparams = rc_optimization(X_train, X_test, y_train, y_test)
    print(hyperparams)

    clf = RidgeClassifier(**hyperparams)
    avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, X, y_truth, True)
    coeffs = clf.coef_

    print(run_ident + " RR (TPE): ", np.round(avg_score, 3), np.round(ppv, 2), np.round(sensitivity, 2), np.round(f_score, 2))

    acc_dict[run_ident + " RR (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score": f_score, "coeffs": coeffs}

    hyperparams = rfc_optimization(X_train, X_test, y_train, y_test)
    print(hyperparams)

    clf = RandomForestClassifier(**hyperparams)
    avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, X, y_truth, True)
    coeffs = clf.feature_importances_

    print(run_ident + " RFC (TPE): ", np.round(avg_score, 3), np.round(ppv, 2), np.round(sensitivity, 2), np.round(f_score, 2))

    acc_dict[run_ident + " RFC (TPE): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score": f_score, "coeffs": coeffs}

    # GridSearch gives similar results but much slower

    # hyperparams = grid_search_rfc(X, y_truth)
    # print(hyperparams)

    # clf = RandomForestClassifier(**hyperparams)
    # avg_score, ppv, sensitivity, f_score, clf = k_folds(clf, X, y_truth, True)
    # coeffs = clf.feature_importances_

    # print(run_ident + " RFC (GS): ", np.round(avg_score, 3), np.round(ppv, 2), np.round(sensitivity, 2), np.round(f_score, 2))

    # acc_dict[run_ident + " RFC (GS): "] = {"score": avg_score, "ppv": ppv, "sensitivity": sensitivity, "f_score": f_score, "coeffs": coeffs}


print("Feature Set, ML Model, Average Score (K-Fold), PPV, Sensitivity, F_score")
for key in acc_dict:
    print(", ".join(key.split(" ")[:-2]), ' ,',
          np.round(acc_dict[key]['score'], 3), ' ,',
          np.round(acc_dict[key]['ppv'], 3), ' ,',
          np.round(acc_dict[key]['sensitivity'], 3), ' ,',
          np.round(acc_dict[key]['f_score'], 3))
