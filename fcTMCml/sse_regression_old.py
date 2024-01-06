from __future__ import print_function
import numpy as np


from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

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


def thd_model_working():
    l2 = 8.812285753542759e-13
    l2 = 8.812285753542759e-1
    l2 = 9.1e-3
    do = 0.0825
    # do = 0.0525
    do = 0.1825
    do = 0.12
    initializer = tf.keras.initializers.HeUniform(seed=0)

    model = Sequential()
    model.add(Dense(64, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2), input_dim=10))
    # model.add(Dense(200, activation="relu", kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2), input_dim=21))
    model.add(Dropout(do))
    # model.add(Dense(64, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2)))
    # model.add(Dropout(do))
    # model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(32, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2)))
    model.add(Dropout(do))
    # model.add(BatchNormalization(momentum=0.99))
    # model.add(Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2)))
    # model.add(Dropout(do))
    # model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(1, activation="linear"))
    lr = 9.1e-4
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, beta_1=0.92, beta_2=0.99),  # decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])
    return model, 256


def model(X_train, Y_train, X_test, Y_test):
    # x_train, y_train, x_test, y_test = data()
    # create model with constraints of hyper-parameters
    activ = {{choice(["relu", "elu"])}} 
    l1 = 0.0  # {{uniform(0, 1)}}
    l2 = {{uniform(0, 10)}}
    drop = {{uniform(0.1, 0.25)}}
    bs = {{randint(140)}} + 64  # {{choice([64, 128])}}#, len(X_train)])}}
    layer1 = {{randint(200)}}
    layer2 = {{randint(200)}}
    neurons = np.sort([layer1 + 64, layer2 + 32])[::-1]
    # momentum = {{choice([0.8, 0.85, 0.9, 0.99])}}
    # momentum = {{uniform(0.85, 0.99)}}
    # layer_size = {{choice([128, 200, 256])}}
    model = Sequential([
        Dense(neurons[0], activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), input_dim=155),
        Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        Dropout(drop),
        # BatchNormalization(momentum=momentum),
        # Dense(64, activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(4, activation="linear"),
    ])

    # compile keras model

    lr = {{choice([1e-3, 1e-2, 1e-4])}}
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])

    # train the model
    # model.fit(X_train, Y_train, epochs=5)
    es = EarlyStopping(monitor='val_mae', mode='min', verbose=0, patience=1000)
    # sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch < 180) else lr * 0.9999)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10000, batch_size=bs, verbose=2, callbacks=[es])

    # evaluate the test performance
    # loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
    # get the highest validation accuracy of the training epochs
    print(history.history.keys())
    print("activation function: ", activ)
    print("l2: ", l2)
    print("dropout: ", drop)
    print("batch size: ", bs)
    print("neurons in dense layers: ", neurons)
    print("learning rate: ", lr)
    validation_mse = np.amin(history.history['val_mse']) 
    print('Best validation acc of epoch:', validation_mse)
    return {'loss': validation_mse, 'status': STATUS_OK, 'model': model}


def plot_linear(test_ranked_by_expected_splitting, res_ranked_by_expected_splitting, rac_name="prod"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(test_ranked_by_expected_splitting, res_ranked_by_expected_splitting)
    ax.set_xlim(0, len(test_ranked_by_expected_splitting))
    ax.set_ylim(0, len(test_ranked_by_expected_splitting))
    ax.set_xlabel(r"Ranking by expected ligand field strength", fontsize=15)
    ax.set_ylabel(r"Ranking by estimated ligand field strength", fontsize=15)
    ax.plot(np.linspace(0, len(test_ranked_by_expected_splitting), 100), np.linspace(0, len(test_ranked_by_expected_splitting), 100), c='k', ls='--')
    ax.tick_params(direction='out')
    ax_r = ax.secondary_yaxis('right')
    ax_t = ax.secondary_xaxis('top')
    ax_r.minorticks_on()
    ax_t.minorticks_on()
    ax.minorticks_on()
    ax_r.tick_params(which='both', direction='in', labelright=False)
    ax_t.tick_params(which='both', direction='in', labeltop=False)
    # plt.legend()
    plt.savefig("{}_lcACs_ligand_strength_pred.pdf".format(rac_name), dpi=300, bbox_inches="tight")
    plt.show()


"""
if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=10000,
                                                 trials=Trials(),
                                                 eval_space=True,
                                                 return_space=True)
    X_train, Y_train, X_test, Y_test = 
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print(best_run)
    print(best_model)
    print(space)
"""


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
            print(hist.history.keys())
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
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
        mae_train = metrics.mean_absolute_error(y_train_test_res, pred_train_res)
        r2 = metrics.r2_score(y_test_res, pred_res)
        mse_n += [mse]
        mse_n_train += [mse_train]
        mae_n += [mae]
        r2_n += [r2]
        if keras_NN:
            clf.set_weights(Wsave)
        print(mae)
        # print(mse)
        # print(mse_train)
        print(mae_train)
        # quit()
    if return_clf:
        return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train), clf
    return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train)


def train_rf_hyperopt(hyperparams, X, y, return_model=False):
    '''
    Train a RF model at given hyperparameters.

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
    clf = RandomForestRegressor(n_estimators=hyperparams["n_estimators"], max_features=hyperparams["max_features"],
                                min_samples_split=hyperparams["min_samples_split"], criterion=hyperparams["criterion"],
                                min_samples_leaf=hyperparams["min_samples_leaf"], random_state=128)
    mse, mae, r2, mse_train, clf = k_folds(clf, X, y, scaler, True)
    if return_model:
        return clf
    return mse


def rf_optimization(X, y):
    '''
    RF hyperparameters optimization with hyperopt.

    Parameters
    ----------
    X_train: np.array
        raining data inputs
    y_train: np.array
        training data targets
    X_val: np.array
        val data inputs
    y_val: np.array
        val data targets

    Returns
    -------
    best: dict
        best hyperparameters.
    '''
    # print("---hyperopt---")
    max_features = ["log2", 'sqrt', 1.0]
    criterion = ["squared_error", "absolute_error"]
    space = {"n_estimators": hp.randint("n_estimators", 10, 100),
             "max_features": hp.choice("max_features", max_features),
             "criterion": hp.choice("criterion", criterion),
             "min_samples_split": hp.loguniform("min_samples_split", np.log(1e-7), np.log(2e-1)),
             "min_samples_leaf": hp.loguniform("min_samples_leaf", np.log(1e-7), np.log(1e-1)),
             }
    objective_func = partial(train_rf_hyperopt,
                             X=X,
                             y=y)
    trials = Trials()
    best = fmin(objective_func,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=100,
                rstate=np.random.default_rng(128)
                )
    best.update({"max_features": max_features[best['max_features']],
                 "criterion": criterion[best['criterion']]})
    print("best_perams: ", best)
    return best


def train_krr_hyperopt(hyperparams, X, y, return_model=False) -> float:
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
    clf = KernelRidge(alpha=hyperparams["alpha"], gamma=hyperparams["gamma"], kernel="rbf")
    mse, mae, r2, mse_train, clf = k_folds(clf, X, y_truth, scaler, True)
    if return_model:
        return clf
    return mae


def krr_optimization(X, y) -> dict:
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
             # "kernel" : hp.choice("kernel", ['polynomial', 'rbf', 'laplacian', 'cosine'])
             }

    objective_func = partial(train_krr_hyperopt,
                             X=X,
                             y=y)
    trials = Trials()
    best = fmin(objective_func,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=40,
                rstate=np.random.default_rng(128)
                )

    print("best_perams: ", best)
    return best


# import data
regression_in_subdir = "regression_raw/"
regression_in_subdir = "regression_rff_selection/"

regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

# make train test split


# best_run, best_model, space = optim.minimize(model=model,
#                                              data=data,
#                                              algo=tpe.suggest,
#                                              max_evals=10000,
#                                              trials=Trials(),
#                                              eval_space=True,
#                                              return_space=True)

def data():
    from fcTMCml.constants import feature_target_dir
    from fcTMCml.tools import load_features
    from sklearn.preprocessing import StandardScaler

    regression_in_subdir = "regression_rff_selection/"
    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")
    X = feature_dict["RAC_regression"]
    scaler = StandardScaler()
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    return X_train, y_train, X_test, y_test

"""
for run_ident in feature_dict:
    X = feature_dict[run_ident]
    scaler = StandardScaler()
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))

    best_run, best_model, space = optim.minimize(model=model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=10000,
                                                 trials=Trials(),
                                                 eval_space=True,
                                                 return_space=True)
    X_train, X_test, y_train, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print(best_run)
    print(best_model)
    print(space)

quit()
"""
#"""
acc_dict = {}
for run_ident in feature_dict:
    X = feature_dict["RAC_cff_regression"]
    # X = feature_dict[run_ident]
    scaler = StandardScaler()
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    working_model, _ = thd_model_working()
    mse, mae, r2, mse_train, clf = k_folds(working_model, X, y_truth, scaler, True, keras_NN=True)
    print(run_ident + " NN (TPE): ", np.round(mse, 3), np.round(mse_train, 3), np.round(mae, 2), np.round(r2, 2))
    coeffs = 0
    acc_dict[run_ident + " NN (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2, "coeffs": coeffs}

    print("Feature Set, ML Model, MSE (K-Fold), MAE, R2")
    for key in acc_dict:
        print(", ".join(key.split(" ")[:-2]), ' ,',
              np.round(acc_dict[key]['mse'], 3), ' ,',
              np.round(acc_dict[key]['mae'], 3), ' ,',
              np.round(acc_dict[key]['r2'], 3), ' ,')

"""
for run_ident in feature_dict:
    # X = feature_dict[run_ident]
    X = feature_dict["RAC_cff_regression"]
    y_truth = regression_targets
    # scaler = StandardScaler()
    # y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, pred)
    mse = metrics.mean_squared_error(y_test, pred, squared=False)
    print(mae, mse)
    pred = clf.predict(X_train)
    mae = metrics.mean_absolute_error(y_train, pred)
    mse = metrics.mean_squared_error(y_train, pred, squared=False)
    print(mae, mse)
"""

acc_dict = {}
for run_ident in feature_dict:
    # X = feature_dict[run_ident]
    X = feature_dict["RAC_cff_regression"]
    scaler = StandardScaler()
    print("std: ", np.std(regression_targets))
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    # X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)

    # print(y_train, y_test)
    hyperparams = krr_optimization(X, y_truth.ravel())
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # hyperparams = {'alpha': 0.539883917167273, 'gamma': 0.15527644872772495}
    # hyperparams = {'alpha': 0.02, 'gamma': 0.1}
    # print(hyperparams)
    clf = KernelRidge(**hyperparams, kernel='rbf')
    mse, mae, r2, mse_train, clf = k_folds(clf, X, y_truth.ravel(), scaler, True)
    coeffs = 0  # clf.dual_coef_

    print(run_ident + " KRR (TPE): ", np.round(mse, 3), np.round(mse_train, 3), np.round(mae, 2), np.round(r2, 2))
    # quit()

    acc_dict[run_ident + " KRR (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2, "coeffs": coeffs}
    continue
    hyperparams = rf_optimization(X, y_truth.ravel())
    clf = RandomForestRegressor(**hyperparams)
    mse, mae, r2, mse_train, clf = k_folds(clf, X, y_truth.ravel(), scaler, True)
    coeffs = 0
    print(run_ident + " RF (TPE): ", np.round(mse, 3), np.round(mse_train, 3), np.round(mae, 2), np.round(r2, 2))
    acc_dict[run_ident + " RF (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2, "coeffs": coeffs}
print("Feature Set, ML Model, MSE (K-Fold), MAE, R2")
for key in acc_dict:
    print(", ".join(key.split(" ")[:-2]), ' ,',
          np.round(acc_dict[key]['mse'], 3), ' ,',
          np.round(acc_dict[key]['mae'], 3), ' ,',
          np.round(acc_dict[key]['r2'], 3), ' ,')
