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

from tqdm.keras import TqdmCallback

import tensorflow.python.keras.backend as K


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
from tensorflow.keras import initializers
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

# quit()

def thd_model_thesis():
    l2 = 9.1e-3
    do = 0.12
    initializer = tf.keras.initializers.HeUniform(seed=0)

    model = Sequential()
    model.add(Dense(64, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2), input_dim=119))
    model.add(Dropout(do))
    model.add(Dense(32, activation="leaky_relu", kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=l2)))
    model.add(Dropout(do))
    model.add(Dense(1, activation="linear"))
    lr = 9.3e-4
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, beta_1=0.92, beta_2=0.99),  # decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])
    return model, 256


def k_folds(clf, X, y, y_scaler=None, return_clf=False, rns=25, keras_NN: bool = False):
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
            es = keras.callbacks.EarlyStopping(monitor='val_mae', mode="min",
                                                     patience=500)

            def scheduler(epoch, lr):
                if epoch > 400:
                    return 1e-4
                elif epoch > 500:
                    return 1 / (epoch * 0.1) * lr
                else:
                    return lr
            lr_sched = keras.callbacks.LearningRateScheduler(scheduler)
            history = clf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000,
                              batch_size=16, verbose=0, callbacks=[es, TqdmCallback(verbose=1)])

        else:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test).reshape(-1, 1)
            pred_train = clf.predict(X_train).reshape(-1, 1)
            if y_scaler is not None:
                y_test_res = y_scaler.inverse_transform(y_test.reshape(-1, 1))
                pred_res = y_scaler.inverse_transform(pred)
                y_train_test_res = y_scaler.inverse_transform(y_train.reshape(-1, 1))
                pred_train_res = y_scaler.inverse_transform(pred_train)
            else:
                y_test_res = y_test
                pred_res = pred
                y_train_test_res = y_train
                pred_train_res = pred_train
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
        if not keras_NN:
            return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train), clf
        else:
            return np.amin(history.history['val_mse']), np.amin(history.history['val_mae']), np.amin(history.history['mse']), np.amin(history.history['mae']), clf
    return np.average(mse_n), np.average(mae_n), np.average(r2_n), np.average(mse_train)

def data():
    # import data
    regression_in_subdir = "regression_rff_selection/"

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict["RAC_cff_regression"]
    scaler = StandardScaler()
    y_truth = scaler.fit_transform(regression_targets.reshape(-1, 1))
    y_truth = regression_targets.reshape(-1, 1)
    # mask = y_truth > -80
    # mask = mask.flatten()
    # print(mask.shape)

    # X = X[mask, :]
    # y_truth = y_truth[mask]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)
    return X_train, Y_train, X_test, Y_test

def model_back(X_train, Y_train, X_test, Y_test):
    activ = {{choice(["leaky_relu", "relu", "elu"])}} 
    l1 = 0.0
    l2 = 0.0#00001 #{{uniform(0, 1)}}
    drop = {{uniform(0.1, 0.25)}}
    bs = 32#{{randint(140)}} + 8  # {{choice([64, 128])}}#, len(X_train)])}}
    layer1 = {{randint(100)}}
    layer2 = {{randint(100)}}
    neurons = np.sort([layer1 + 64, layer2 + 32])[::-1]
    momentum = {{uniform(0.85, 0.99)}}
    model = Sequential([
        Dense(neurons[0], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), input_dim=119),
        Dropout(drop),
        BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        Dropout(drop),
        BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        BatchNormalization(momentum=momentum),
        Dense(64, activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(1, activation="linear"),
    ])

    # compile keras model

    lr = {{loguniform(-5, -2)}}
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])

    # train the model
    # model.fit(X_train, Y_train, epochs=5)
    es = EarlyStopping(monitor='val_mae', mode='min', verbose=0, patience=100)
    # sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch < 180) else lr * 0.9999)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=bs, verbose=0, callbacks=[es, TqdmCallback(verbose=1)])

    # evaluate the test performance
    # loss, accuracy = model.evaluate(X_test,  Y_test, verbose=2)
    # get the highest validation accuracy of the training epochs
    print(history.history.keys())
    print("activation function: ", activ)
    print("l2: ", l2)
    print("dropout: ", drop)
    print("batch size: ", bs)
    print("neurons in dense layers: ", neurons)
    print("learning rate: ", lr)
    validation_mse = np.amin(history.history['val_mae']) 
    print('Best validation acc of epoch:', validation_mse)
    return {'loss': validation_mse, 'status': STATUS_OK, 'model': model}


def model(X_train, Y_train, X_test, Y_test):
    activ = "elu" 
    l1 = 0.0
    l2 = {{uniform(0, 1)}}
    drop = {{uniform(0.1, 0.25)}}
    bs = 32
    layer1 = 256
    layer2 = 128
    neurons = np.sort([layer1, layer2])[::-1]
    momentum = {{uniform(0.85, 0.99)}}
    model = Sequential([
        Dense(neurons[0], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), input_dim=119),
        Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        Dense(64, activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        
        # Dropout(drop),
        # BatchNormalization(momentum=momentum),
        # Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
        #       kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # BatchNormalization(momentum=momentum),
        # Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
        #       kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # BatchNormalization(momentum=momentum),
        # Dense(64, activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(1, activation="linear"),
    ])

    # compile keras model

    lr = {{loguniform(-5, -2)}}
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, decay=lr / 200),
                  metrics=['mse', 'mae', 'mape'])

    # train the model
    # model.fit(X_train, Y_train, epochs=5)
    es = EarlyStopping(monitor='val_mae', mode='min', verbose=0, patience=100)
    # sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch < 180) else lr * 0.9999)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=bs, verbose=0, callbacks=[es, TqdmCallback(verbose=1)])

    # evaluate the test performance
    # loss, accuracy = model.evaluate(X_test,  Y_test, verbose=2)
    # get the highest validation accuracy of the training epochs
    print(history.history.keys())
    print("activation function: ", activ)
    print("l2: ", l2)
    print("dropout: ", drop)
    print("batch size: ", bs)
    print("neurons in dense layers: ", neurons)
    print("learning rate: ", lr)
    validation_mse = np.amin(history.history['val_mae']) 
    print('Best validation acc of epoch:', validation_mse)
    return {'loss': validation_mse, 'status': STATUS_OK, 'model': model}




def model_hyperparams(X_train, Y_train, X_test, Y_test, hyperparams, return_model=False):
    activ = ["leaky_relu", "relu", "elu"][int(hyperparams['activ'])]
    l1 = 0.0
    l2 = hyperparams['l2']
    drop = hyperparams['drop']
    bs = hyperparams['bs']
    layer1 = hyperparams['layer1']
    layer2 = hyperparams['layer1_1']
    neurons = np.sort([layer1 + 64, layer2 + 32])[::-1]
    # momentum = {{choice([0.8, 0.85, 0.9, 0.99])}}
    momentum = hyperparams['momentum']
    # layer_size = {{choice([128, 200, 256])}}
    model = Sequential([
        Dense(neurons[0], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), input_dim=119),
        Dropout(drop),
        BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # Dropout(drop),
        BatchNormalization(momentum=momentum),
        Dense(neurons[1], activation=activ, kernel_initializer=initializers.GlorotNormal(seed=821),
              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # BatchNormalization(momentum=momentum),
        # Dense(64, activation=activ, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
        # Dropout(drop),
        # BatchNormalization(momentum=momentum),
        Dense(1, activation="linear"),
    ])

    # compile keras model

    lr = hyperparams['lr']
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr, decay=lr / 200, beta_1 = 0.98, beta_2 = 0.99 ),
                  metrics=['mse', 'mae', 'mape'])

    # train the model
    # model.fit(X_train, Y_train, epochs=5)
    es = EarlyStopping(monitor='val_mae', mode='min', verbose=0, patience=1000)
    # sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch < 180) else lr * 0.9999)
    if return_model:
        return model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10000, batch_size=bs, verbose=1, callbacks=[es])

    # evaluate the test performance
    # loss, accuracy = model.evaluate(X_test,  Y_test, verbose=2)
    # get the highest validation accuracy of the training epochs
    validation_mse = np.amin(history.history['val_mae']) 
    print('Best validation acc of epoch:', validation_mse)
    return {'loss': validation_mse, 'status': STATUS_OK, 'model': model}


######
# NN #
######
keras.utils.set_random_seed(821)
# '''
if __name__ == "__main__":
    import gc; gc.collect()

    with K.get_session(): ## TF session
        best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=200,
                                              rseed=128,
                                              verbose=False,
                                              trials=Trials())
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_test, Y_test, verbose=0))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)

# '''
quit()
# """
acc_dict = {}
# import data
regression_in_subdir = "regression_rff_selection/"

regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

X = feature_dict["RAC_cff_regression"]
run_ident = "RAC_cff_regression"

y_truth = np.array(regression_targets)
X_train, X_test, y_train, y_test = train_test_split(X, y_truth, test_size=0.2, random_state=128)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
hyperparams_25_01 = {'activ': 0, 'bs': 64, 'bs_1': 0, 'drop': 0.12395172243676891, 'l2': 0.001940783149560846, 'layer1': 182, 'layer1_1': 31, 'layer_size': 1, 'lr': 1e-3, 'momentum': 0.902178111485586, 'momentum_1': 0.902178111485586}
hyperparams_27_01 = {'activ': 1, 'bs': 16, 'bs_1': 1, 'drop': 0.10036071901785594, 'l2': 0.6625126585770287, 'layer1': 54, 'layer1_1': 108, 'lr': 0.012845109781685538, 'momentum': 0.9548741374370129}
       
hyperparams = hyperparams_25_01
working_model = model_hyperparams(X_train, y_train, X_test, y_test, hyperparams, return_model=True)
# working_model, _ = thd_model_thesis()
# print(scaler.inverse_transform(np.array([results['loss']]).reshape(1, -1)))
mse, mae, mse_train, mae_train, clf = k_folds(working_model, X, y_truth, return_clf=True, keras_NN=True)
print(run_ident + " NN (TPE): ", np.round(mae_train, 3), np.round(mae, 3), np.round(mse_train, 2), np.round(mse, 2))
coeffs = 0
acc_dict[run_ident + " NN (TPE): "] = {"mse": mse, "mse_train": mse_train, "mae": mae, "r2": r2, "coeffs": coeffs}

print("Feature Set, ML Model, MSE (K-Fold), MSE train, MAE, R2")
for key in acc_dict:
    print(", ".join(key.split(" ")[:-2]), ' ,',
            np.round(acc_dict[key]['mse'], 3), ' ,',
            np.round(acc_dict[key]['mse_train'], 3), ' ,',
            np.round(acc_dict[key]['mae'], 3), ' ,',
            np.round(acc_dict[key]['r2'], 3), ' ,')

# """
