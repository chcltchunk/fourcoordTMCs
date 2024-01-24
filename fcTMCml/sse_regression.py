from __future__ import print_function
import numpy as np


from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split

from hyperopt import hp, tpe, fmin, Trials
from functools import partial


from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json, load_model
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hyperas.distributions import choice, uniform, randint
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def thd_model_thesis():
    l2 = 9.1e-3
    do = 0.12
    initializer = tf.keras.initializers.HeUniform(seed=0)

    model = Sequential()
    model.add(Dense(64, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2), input_dim=17))
    model.add(Dropout(do))
    model.add(Dense(32, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2)))
    model.add(Dropout(do))
    model.add(Dense(1, activation="linear"))
    lr = 9.3e-4
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, beta_1=0.92, beta_2=0.99),  # decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])
    return model, 256


def k_folds(clf, X, y, y_scaler, return_clf=False, rns=25, keras_NN: bool = False):
    kf = KFold(n_splits=10, shuffle=True, random_state=128)
    kf.get_n_splits(X)
    mse_n = []
    mse_n_train = []
    mae_n = []
    r2_n = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if keras_NN:
            Wsave = clf.get_weights()
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode="min",
                                                     patience=500)

            def scheduler(epoch, lr):
                if epoch > 400:
                    return 1e-4
                elif epoch > 500:
                    return 1 / (epoch * 0.1) * lr
                else:
                    return lr
            lr_sched = keras.callbacks.LearningRateScheduler(scheduler)
            hist = clf.fit(X_train, y_train, validation_split=0.1, epochs=10000, batch_size=64, callbacks=[callback])
        else:
            clf.fit(X_train, y_train)
        pred = clf.predict(X_test).reshape(-1, 1)
        pred_train = clf.predict(X_train).reshape(-1, 1)
        y_test_res = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        pred_res = y_scaler.inverse_transform(pred)
        y_train_test_res = y_scaler.inverse_transform(y_train.reshape(-1, 1))
        pred_train_res = y_scaler.inverse_transform(pred_train)
        mse = metrics.mean_squared_error(y_test_res, pred_res, squared=False)
        mse_train = metrics.mean_squared_error(y_train_test_res, pred_train_res, squared=False)
        mae = metrics.mean_absolute_error(y_test_res, pred_res)
        # mae_train = metrics.mean_absolute_error(y_train_test_res, pred_train_res)
        r2 = metrics.r2_score(y_test_res, pred_res)
        mse_n += [mse]
        mse_n_train += [mse_train]
        mae_n += [mae]
        r2_n += [r2]
        if keras_NN:
            clf.set_weights(Wsave)
    if return_clf:
        return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train), clf
    return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train)


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
    # krr = KernelRidge(alpha=hyperparams["alpha"], gamma=hyperparams["gamma"], kernel=hyperparams['kernel'])
    krr = KernelRidge(alpha=hyperparams["alpha"], gamma=hyperparams["gamma"], kernel="rbf")
    krr.fit(X_train, y_train)
    if return_model:
        return krr
    y_pred = krr.predict(X_val)
    mse = metrics.mean_squared_error(y_val, y_pred, squared=False)
    # mae = metrics.mean_absolute_error(y_val, y_pred)
    return mse


def krr_optimization(X_train: np.array, X_val: np.array, y_train: np.array, y_val: np.array) -> dict:
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
    space = {"alpha": hp.loguniform("alpha", -8, 1),
             "gamma": hp.loguniform("gamma", -12, 1),
             }

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
    kf = KFold(n_splits=10, shuffle=True, random_state=185)
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


# import data
regression_in_subdir = "regression_rff_selection/"

regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")


#######
# KRR #
#######
# """
acc_dict = {}
for run_ident in feature_dict:
    X = feature_dict[run_ident]
    scaler = StandardScaler()
    print("std: ", np.std(regression_targets))
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    hyperparams = krr_optimization(X_train, X_test, y_train, y_test)
    print(hyperparams)
    clf = KernelRidge(**hyperparams, kernel='rbf')
    mse, mae, r2, mse_train, clf = k_folds(clf, X, y_truth.ravel(), scaler, True)

    print(run_ident + " KRR (TPE): ", np.round(mse, 3), np.round(mse_train, 3), np.round(mae, 2), np.round(r2, 2))

    acc_dict[run_ident + " KRR (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2}


print("Feature Set, ML Model, MSE (K-Fold), MAE, R2")
for key in acc_dict:
    print(", ".join(key.split(" ")[:-2]), ' ,',
          np.round(acc_dict[key]['mse'], 3), ' ,',
          np.round(acc_dict[key]['mae'], 3), ' ,',
          np.round(acc_dict[key]['r2'], 3), ' ,')
# """
######
# NN #
######
acc_dict = {}
for run_ident in feature_dict:
    if run_ident in ["MCDLF_regression", "MCDLF_cff_regression"]:
        continue
    X = feature_dict[run_ident]
    scaler = StandardScaler()
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    working_model, _ = thd_model_thesis()
    mse, mae, r2, mse_train, clf = k_folds(working_model, X, y_truth, scaler, True, keras_NN=True)
    print(run_ident + " NN (TPE): ", np.round(mse, 3), np.round(mse_train, 3), np.round(mae, 2), np.round(r2, 2))
    coeffs = 0
    acc_dict[run_ident + " NN (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2, "coeffs": coeffs}

    print("Feature Set, ML Model, MSE (K-Fold), MSE train, MAE, R2")
    for key in acc_dict:
        print(", ".join(key.split(" ")[:-2]), ' ,',
              np.round(acc_dict[key]['mse'], 3), ' ,',
              np.round(acc_dict[key]['mse_train'], 3), ' ,',
              np.round(acc_dict[key]['mae'], 3), ' ,',
              np.round(acc_dict[key]['r2'], 3), ' ,')
