from __future__ import print_function
import numpy as np



from sklearn.model_selection import train_test_split

from hyperopt import hp, tpe, fmin, Trials, space_eval

from tqdm.keras import TqdmCallback


from sklearn.model_selection import KFold

import tensorflow.python.keras.backend as K


from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam  
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json, load_model
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hyperas.distributions import choice, uniform, randint
from hyperopt import Trials, STATUS_OK, tpe

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


space = {
    'optimizer': hp.choice('optimizer', [Adam]),
    'epochs': hp.choice('epochs', [150, 300, 800]),
    'batch_size': hp.choice('batch_size', [16, 32, 256]),  # [16, 64]),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'num_layers': hp.choice('layers', [2, 3]),
    'layers': [
        {
            'type': 'dense',
            'units': hp.choice('dense_units', [64, 128]),
            'activation': hp.choice('dense_activation', ['leaky_relu', 'elu']),
            'l2_reg': hp.loguniform('dense_l2_reg', -3, 0),
            'batch_norm' : hp.uniform('batch_norm', 0.85, 0.99),
            'dropout': hp.uniform('dense_dropout', 0, 0.5)
        },
        {
            'type': 'dense',
            'units': hp.choice('dense_units_1', [32, 64]),
            'activation': hp.choice('dense_activation_1', ['leaky_relu', 'elu']),
            'l2_reg': hp.loguniform('dense_l2_reg_1', -3, 0),
            'batch_norm' : hp.uniform('batch_norm_1', 0.85, 0.99),
            'dropout': hp.uniform('dense_dropout_1', 0, 0.5)
        },
        {
            'type': 'dense',
            'units': hp.choice('dense_units_2', [32, 64]),
            'activation': hp.choice('dense_activation_2', ['leaky_relu', 'elu']),
            'l2_reg': hp.loguniform('dense_l2_reg_2', -3, 0),
            'batch_norm' : hp.uniform('batch_norm_2', 0.85, 0.99),
            'dropout': hp.uniform('dense_dropout_2', 0, 0.5)
        },
        {
            'type': 'dense',
            'units': hp.choice('dense_units_3', [32, 64]),
            'activation': hp.choice('dense_activation_3', ['leaky_relu', 'elu']),
            'l2_reg': hp.loguniform('dense_l2_reg_3', -3, 0),
            'batch_norm' : hp.uniform('batch_norm_3', 0.85, 0.99),
            'dropout': hp.uniform('dense_dropout_3', 0, 0.5)
        },
        # {
        #     'type': 'dense',
        #     'units': hp.choice('dense_units_2', [16, 32]),
        #     'activation': hp.choice('dense_activation_2', ['leaky_relu', 'elu']),
        #     'l2_reg': hp.uniform('dense_l2_reg_2', 0, 0.01),
        #     # 'dropout': hp.uniform('dense_dropout_2', 0, 0.5)
        # }
    ]
}


def create_model(params):
    model = Sequential()
    print(params)
    # print(params['layers'])
    for i in range(params['num_layers']):
        layer_params = params['layers'][i]
        if layer_params['type'] == 'dense':
            model.add(Dense(units=layer_params['units'],
                            activation=layer_params['activation'],
                            kernel_regularizer=regularizers.L2(layer_params['l2_reg']),
                            kernel_initializer=initializers.GlorotNormal(seed=821)))
            if 'dropout' in layer_params and i < params['num_layers'] - 1:
                model.add(Dropout(rate=layer_params['dropout']))
                model.add(BatchNormalization(momentum=layer_params['batch_norm']))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=params['optimizer'](learning_rate=params['learning_rate']), loss='mse', metrics=['mae', 'mse'])
    # print(model.summary())
    return model


def training(params, key: str = "RAC_cff_regression", return_history: bool = False):
    model = create_model(params)
    # Train your model and return the value to minimize
    X_train, Y_train, X_test, Y_test = data()
    es = EarlyStopping(monitor='mse', mode='auto', verbose=0, patience=100)
    # print(X_train.shape, Y_train.shape)
    # quit()
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=params['epochs'], batch_size=params['batch_size'], verbose=1, callbacks=[es])
    loss = history.history['val_mse'][-1]
    # plot_model(model, "model.png")
    if return_history:
        return history
    return loss


def data(key: str = "RAC_cff_regression"):
    # import data
    regression_in_subdir = "regression_rff_selection/"

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

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict[key]
    y_truth = regression_targets.reshape(-1, 1)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf.get_n_splits(X)
    loss_n = []
    loss_train_n = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_truth[train_index], y_truth[test_index]

        model = create_model(params)
        es = EarlyStopping(monitor='mse', mode='auto', verbose=0, patience=100)
    
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=params['epochs'],
                            batch_size=params['batch_size'], verbose=1, callbacks=[es])
        loss = history.history['val_mae'][-1]
        loss_n += [loss]
        loss = history.history['mae'][-1]
        loss_train_n += [loss]
    return np.average(loss_train_n), np.average(loss_n)


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
        best = fmin(fn=objective_kfold_rac_cff, space=space, algo=tpe.suggest, max_evals=400, show_progressbar=True)
        print("Best hyperparameters:", best)
        # best = {'batch_norm': 0.9346860946367082, 'batch_norm_1': 0.9702570581854131, 'batch_norm_2': 0.9249498413567074, 'batch_norm_3': 0.9284334406266748, 'batch_size': 0, 'dense_activation': 1, 'dense_activation_1': 0, 'dense_activation_2': 1, 'dense_activation_3': 1, 'dense_dropout': 0.1810803373583213, 'dense_dropout_1': 0.49460993462692243, 'dense_dropout_2': 0.15088107232837267, 'dense_dropout_3': 0.42973749770328307, 'dense_l2_reg': 0.10607279197122027, 'dense_l2_reg_1': 0.06232231061813992, 'dense_l2_reg_2': 0.1140963417907163, 'dense_l2_reg_3': 0.9067263774921112, 'dense_units': 1, 'dense_units_1': 0, 'dense_units_2': 1, 'dense_units_3': 0, 'epochs': 0, 'layers': 0, 'learning_rate': 0.090201604561948, 'optimizer': 1}
        best_hp = space_eval(space, best)
        history = training(best_hp, key="RAC_cff_regression", return_history=True)
        train_loss, val_loss = history.history["mae"][-1], history.history["val_mae"][-1]
        np.save("train_mae_rac_cff.npy", history.history["mae"])
        np.save("val_mae_rac_cff.npy", history.history["val_mae"])
        result_dict["RAC_cff_regression"] = [train_loss, val_loss, best_hp]
        # train_loss, val_loss = run_kfold(best_hp)
        # print(train_loss, val_loss)
        # result_dict["RAC_cff_regression_kfold"] = [train_loss, val_loss, best_hp]
        
        best = fmin(fn=objective_kfold_rac, space=space, algo=tpe.suggest, max_evals=400, show_progressbar=True)
        print("Best hyperparameters:", best)
        # best = {'batch_norm': 0.9346860946367082, 'batch_norm_1': 0.9702570581854131, 'batch_norm_2': 0.9249498413567074, 'batch_norm_3': 0.9284334406266748, 'batch_size': 0, 'dense_activation': 1, 'dense_activation_1': 0, 'dense_activation_2': 1, 'dense_activation_3': 1, 'dense_dropout': 0.1810803373583213, 'dense_dropout_1': 0.49460993462692243, 'dense_dropout_2': 0.15088107232837267, 'dense_dropout_3': 0.42973749770328307, 'dense_l2_reg': 0.10607279197122027, 'dense_l2_reg_1': 0.06232231061813992, 'dense_l2_reg_2': 0.1140963417907163, 'dense_l2_reg_3': 0.9067263774921112, 'dense_units': 1, 'dense_units_1': 0, 'dense_units_2': 1, 'dense_units_3': 0, 'epochs': 0, 'layers': 0, 'learning_rate': 0.090201604561948, 'optimizer': 1}
        best_hp = space_eval(space, best)
        history = training(best_hp, key="RAC_regression", return_history=True)
        train_loss, val_loss = history.history["mae"][-1], history.history["val_mae"][-1]
        np.save("train_mae_rac.npy", history.history["mae"])
        np.save("val_mae_rac.npy", history.history["val_mae"])
        result_dict["RAC_regression"] = [train_loss, val_loss, best_hp]
        # train_loss, val_loss = run_kfold(best_hp)
        # print(train_loss, val_loss)
        # result_dict["RAC_regression_kfold"] = [train_loss, val_loss, best_hp]
    print(result_dict)
        
