from __future__ import print_function
import numpy as np

from sklearn.model_selection import train_test_split

from hyperopt import hp
# uncomment for hyperparameter training
# from hyperopt import tpe, fmin, Trials, space_eval

from sklearn.model_selection import KFold

import tensorflow.python.keras.backend as K

from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import initializers, regularizers


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


space = {
    "hidden_units": hp.choice("hidden_units", [[64, 64], [128, 128], [256, 256]]),
    "l2_reg": hp.quniform("l2_reg", -5, 0, 1),
    "dropout": hp.quniform("dropout", 0.0, 0.6, 0.1),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "learning_rate": hp.loguniform('learning_rate', -5, -1),
    "activation": hp.choice('dense_activation', ['leaky_relu', 'softplus'])
}


def create_model(params):
    model = Sequential()
    print(params)
    # print(params['layers'])

    model.add(Dense(units=params['hidden_units'][0],
                    activation=params['activation'],
                    kernel_regularizer=regularizers.L2(params['l2_reg']),
                    kernel_initializer=initializers.GlorotNormal(seed=821)))
    model.add(Dropout(rate=params['dropout']))
    # model.add(BatchNormalization(momentum=params['batch_norm']))
    model.add(Dense(units=params['hidden_units'][1],
                    activation=params['activation'],
                    kernel_regularizer=regularizers.L2(params['l2_reg']),
                    kernel_initializer=initializers.GlorotNormal(seed=821)))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse', metrics=['mae', 'mse'])
    # print(model.summary())
    return model


def training(params, key: str = "RAC_cff_regression", return_history: bool = False):
    model = create_model(params)
    # Train your model and return the value to minimize
    X_train, Y_train, X_test, Y_test = data()
    es = EarlyStopping(monitor='mse', mode='auto', verbose=0, patience=100)
    # print(X_train.shape, Y_train.shape)
    # quit()
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=params['batch_size'], verbose=0,
                        callbacks=[es])
    loss = history.history['val_mse'][-1]
    if return_history:
        return history
    return loss


def data(key: str = "RAC_cff_regression"):
    # import data
    regression_in_subdir = "regression_rff_selection/"
    regression_in_subdir = "regression_raw/"

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict[key]
    y_truth = regression_targets
    # scaler = StandardScaler()
    # y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    # # y_truth = regression_targets.reshape(-1, 1)
    # mask = y_truth > -80
    # mask = mask.flatten()
    # print(mask.shape)

    # X = X[mask, :]
    # y_truth = y_truth[mask]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_truth, test_size=0.1, random_state=821)
    return X_train, Y_train.reshape(-1, 1), X_test, Y_test.reshape(-1, 1)


def run_kfold(params, key: str = "RAC_cff_regression"):
    # import data
    regression_in_subdir = "regression_rff_selection/"
    regression_in_subdir = "regression_raw/"

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict[key]
    y_truth = regression_targets.reshape(-1, 1)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(X)
    mae_train_n = []
    mae_test_n = []
    mse_train_n = []
    mse_test_n = []
    r2_n = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_truth[train_index], y_truth[test_index]

        model = create_model(params)
        es = EarlyStopping(monitor='mse', mode='auto', verbose=0, patience=100)

        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000,
                            batch_size=params['batch_size'], verbose=0, callbacks=[es])
        mae_train, mae_test = history.history['mae'][-1], history.history['val_mae'][-1]
        mae_train_n += [mae_train]
        mae_test_n += [mae_test]
        mse_train, mse_test = history.history["mse"][-1], history.history["val_mse"][-1]
        mse_train_n += [mse_train]
        mse_test_n += [mse_test]
       
    return np.average(mae_train_n), np.average(mae_test_n), np.average(mse_train_n), np.average(mse_test_n)


def objective_rac_cff(params):
    return training(params)


def objective_rac(params):
    return training(params, key="RAC_regression")


def objective_kfold_rac_cff(params):
    _, b = run_kfold(params)
    return b


def objective_kfold_rac(params):
    _, b = run_kfold(params, key="RAC_regression")
    return b


######
# NN #
######
keras.utils.set_random_seed(821)
if __name__ == "__main__":
    import gc
    gc.collect()
    result_dict = {}
    with K.get_session():
        # uncomment for hyperparameter training
        # best = fmin(fn=objective_rac_cff, space=space, algo=tpe.suggest, max_evals=120, show_progressbar=True)
        # best_hp = space_eval(space, best)
        best_hp = {'activation': 'softplus', 'batch_size': 32, 'dropout': 0.3, 'hidden_units': (256, 256), 'l2_reg': 0.0, 'learning_rate': 0.01654324497496395}
        best_hp = {'activation': 'softplus', 'batch_size': 32, 'dropout': 0.3, 'hidden_units': (256, 256), 'l2_reg': 0.0, 'learning_rate': 0.0165}
        print("Best hyperparameters:", best_hp)
        # TODO: k-fold
        history = training(best_hp, key="RAC_cff_regression", return_history=True)
        mae_train, mae_test = history.history["mae"][-1], history.history["val_mae"][-1]
        mse_train, mse_test = history.history["mse"][-1], history.history["val_mse"][-1]
        # r2 = history.history['r2_score'][-1]
        # TODO: plot learning curves final version
        np.save("train_mae_rac_cff.npy", history.history["mae"])
        np.save("val_mae_rac_cff.npy", history.history["val_mae"])
        result_dict["RAC_cff_regression"] = [mae_train, mae_test, mse_train, mse_test, best_hp]
        mae_train, mae_test, mse_train, mse_test = run_kfold(best_hp)
        result_dict["RAC_cff_regression_kfold"] = [mae_train, mae_test, mse_train, mse_test, best_hp]
       
        # uncomment for hyperparameter training
        # best = fmin(fn=objective_rac, space=space, algo=tpe.suggest, max_evals=120, show_progressbar=True)
        # print("Best hyperparameters:", best)
        # best_hp = space_eval(space, best)
        best_hp = {'activation': 'leaky_relu', 'batch_size': 128, 'dropout': 0.3, 'hidden_units': (128, 128), 'l2_reg': 0.0, 'learning_rate': 0.038983468378456794}
        best_hp = {'activation': 'leaky_relu', 'batch_size': 128, 'dropout': 0.3, 'hidden_units': (128, 128), 'l2_reg': 0.0, 'learning_rate': 0.039}
        history = training(best_hp, key="RAC_regression", return_history=True)
        mae_train, mae_test = history.history["mae"][-1], history.history["val_mae"][-1]
        mse_train, mse_test = history.history["mse"][-1], history.history["val_mse"][-1]
        # r2 = history.history['r2_score'][-1]
        np.save("train_mae_rac.npy", history.history["mae"])
        np.save("val_mae_rac.npy", history.history["val_mae"])
        result_dict["RAC_regression"] = [mae_train, mae_test, mse_train, mse_test, best_hp]
        mae_train, mae_test, mse_train, mse_test = run_kfold(best_hp)
        result_dict["RAC_regression_kfold"] = [mae_train, mae_test, mse_train, mse_test, best_hp]
    print(result_dict)
    
