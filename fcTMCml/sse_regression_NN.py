from __future__ import print_function
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2_sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from numpy.random import default_rng

from hyperopt import hp
# uncomment for hyperparameter training
from hyperopt import tpe, fmin, Trials, space_eval, rand

import os

from sklearn.model_selection import KFold

import tensorflow.keras.backend as K

from fcTMCml.constants import feature_target_dir
from fcTMCml.tools import load_features, plot_learning_curve_NN, plot_parity_plot, make_dir

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsolutePercentageError

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"



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


random_seed = 423890532 # 821

space = {
    "hidden_units": hp.choice("hidden_units", [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]),
    # "hidden_units": hp.choice("hidden_units", [[16, 16], [32, 32], [32, 64], [64, 32], [64, 64], [64, 128], [128, 64], [128, 128],
    #                                            [64, 64, 64], [128, 128, 128], [16, 16, 16], [32, 32, 32]]),
    # "l2_reg": hp.loguniform("l2_reg", -13.8, -4.6),
    # "l2_reg_1": hp.loguniform("l2_reg_1", -13.8, 0),
    # "l2_reg_2": hp.loguniform("l2_reg_2", -13.8, 0),
    "l2_reg_1": hp.loguniform("l2_reg_1", -10, -1),
    "l2_reg_2": hp.loguniform("l2_reg_2", -10, -1),
    "dropout": hp.quniform("dropout", 0.01, 0.3, 0.01),
    # "beta_1": hp.quniform("beta_1", 0.6, 0.99, 0.01),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "learning_rate": hp.loguniform('learning_rate', -12, -6),
    # "learning_rate": hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    # "lr_schedule_factor": hp.quniform("lr_schedule_factor", 0.2, 0.5, 0.01),
    "activation": hp.choice('dense_activation', ['leaky_relu', 'softplus', 'elu', 'gelu']),
}


def create_model(params):
    # print(params)
    model = Sequential()
    # print(params['layers'])

    model.add(Dense(units=params['hidden_units'][0],
                    activation=params['activation'],
                    kernel_regularizer=regularizers.L2(params['l2_reg_1']),
                    kernel_initializer=initializers.GlorotNormal(seed=random_seed)))
    model.add(Dropout(rate=float(params['dropout'])))
    # model.add(BatchNormalization(momentum=params['batch_norm']))
    model.add(Dense(units=params['hidden_units'][1],
                    activation=params['activation'],
                    kernel_regularizer=regularizers.L2(params['l2_reg_2']),
                    kernel_initializer=initializers.GlorotNormal(seed=random_seed)))

    if len(params["hidden_units"]) > 2:
        # model.add(Dropout(rate=float(params['dropout'])))
        model.add(Dense(units=params['hidden_units'][2],
                        activation=params['activation'],
                        kernel_regularizer=regularizers.L2(params['l2_reg_2']),
                        kernel_initializer=initializers.GlorotNormal(seed=random_seed)))

    model.add(Dense(1, activation='linear'))
    # model.compile(optimizer=Adam(learning_rate=params['learning_rate'], beta_1=params["beta_1"]), loss='mse', metrics=['mae', 'mse', r2_score])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse', metrics=['mae', 'mse', r2_score, MeanAbsolutePercentageError()])
    # print(model.summary())
    return model


def training(params, key: str = "RAC_cff_regression", return_history: bool = False):
    # print(params)
    tf.keras.utils.set_random_seed(random_seed)
    model = create_model(params)
    # Train your model and return the value to minimize
    X_train, Y_train, X_test, Y_test = data(key=key)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=20, restore_best_weights=True)

    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=params["lr_schedule_factor"], patience=10, min_lr=1e-5)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=params['batch_size'], verbose=0,
                        callbacks=[es], shuffle=True)
    loss = min(history.history['val_loss'])
    if return_history:
        return history
    return loss


def data(key: str = "RAC_cff_regression"):
    # import data
    regression_in_subdir = "regression_rff_selection/"
    # regression_in_subdir = "regression_raw/"

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict[key]
    y_truth = regression_targets
    # # y_truth = regression_targets.reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_truth, test_size=0.2, random_state=random_seed)
    scaler = StandardScaler()
    Y_train = scaler.fit_transform(Y_train.reshape(-1, 1))
    Y_test = scaler.transform(Y_test.reshape(-1, 1))
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    return X_train, Y_train, X_test, Y_test


# https://stackoverflow.com/questions/45250100/kerasregressor-coefficient-of-determination-r2-score
def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def run_kfold(params, key: str = "RAC_cff_regression", folds=5, patience=20):
    # import data
    regression_in_subdir = "regression_rff_selection/"
    # regression_in_subdir = "regression_raw/"

    regression_targets, feature_dict, feature_names_dict = load_features(feature_target_dir, regression_in_subdir, "regression")

    X = feature_dict[key]
    # scaler = StandardScaler()
    y_truth = regression_targets.reshape(-1, 1)
    # y_truth = scaler.fit_transform(y_truth)

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kf.get_n_splits(X)
    mae_train_n = []
    mae_test_n = []
    mse_train_n = []
    mse_test_n = []
    mape_train_n = []
    mape_test_n = []
    r2_test_n = []
    y_truth_n = []
    y_predict_n = []
    y_data_set_index = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y_truth[train_index], y_truth[test_index]
        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y_train)
        Y_test = scaler.transform(Y_test)
        # scaler_n += [scaler]
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        tf.keras.utils.set_random_seed(random_seed)
        model = create_model(params)
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=patience)
        # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=params["lr_schedule_factor"], patience=10)

        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150,
                            batch_size=params['batch_size'], verbose=0, callbacks=[es], shuffle=True)
        y_predict_n_curr = list(scaler.inverse_transform(model.predict(X_test, verbose=0)))
        y_predict_n_curr_train = list(scaler.inverse_transform(model.predict(X_train, verbose=0)))
        # y_predict_n_curr = list(model.predict(X_test, verbose=0))
        # y_predict_n_curr_train = list(model.predict(X_train, verbose=0))
        y_predict_n += [*y_predict_n_curr]
        y_truth_n_curr = list(scaler.inverse_transform(Y_test))
        y_truth_n_curr_train = list(scaler.inverse_transform(Y_train))
        # y_truth_n_curr = list(Y_test)
        # y_truth_n_curr_train = list(Y_train)

        y_truth_n += [*y_truth_n_curr]
        y_data_set_index += [*list(test_index)]
        mae_train = history.history['mae'][-1]
        mae_test = history.history['val_mae'][-1]
        r2_test = history.history['val_r2_score'][-1]
        mape_train = history.history['mean_absolute_percentage_error'][-1]
        mape_test = history.history['val_mean_absolute_percentage_error'][-1]
        mse_train = history.history["mse"][-1]
        mse_test = history.history["val_mse"][-1]
        mae_train = mean_absolute_error(y_truth_n_curr_train, y_predict_n_curr_train)
        mae_test = mean_absolute_error(y_truth_n_curr, y_predict_n_curr)
        r2_test = r2_sklearn(y_truth_n_curr, y_predict_n_curr)
        mape_train = mean_absolute_percentage_error(y_truth_n_curr_train, y_predict_n_curr_train)
        mape_test = mean_absolute_percentage_error(y_truth_n_curr, y_predict_n_curr)
        mse_train = mean_squared_error(y_truth_n_curr_train, y_predict_n_curr_train)
        mse_test = mean_squared_error(y_truth_n_curr, y_predict_n_curr)
        mape_train_n += [mape_train]
        mape_test_n += [mape_test]
        mae_train_n += [mae_train]
        mae_test_n += [mae_test]
        mse_train_n += [mse_train]
        mse_test_n += [mse_test]
        r2_test_n += [r2_test]
    y_pred_full = np.empty_like(y_truth)
    y_true_full = np.empty_like(y_truth)

    for idx, pred, true in zip(y_data_set_index, y_predict_n, y_truth_n):
        y_pred_full[idx] = pred
        y_true_full[idx] = true

    y_predict_n = y_pred_full
    y_truth_n = y_true_full

    # y_predict_n = scaler.inverse_transform(y_pred_full)
    # y_truth_n = scaler.inverse_transform(y_true_full)
    # print(y_truth_n)
    # print(y_predict_n)

    return np.average(mae_train_n), np.average(mae_test_n), np.average(mse_train_n), np.average(mse_test_n), np.average(r2_test_n), np.average(mape_train_n), np.average(mape_test_n), y_predict_n, y_truth_n


def objective_rac_cff(params):
    K.clear_session()
    return training(params)


def objective_rac(params):
    K.clear_session()
    return training(params, key="RAC_regression")


def objective_kfold_rac_cff(params):
    K.clear_session()
    _, _, _, b, c, _, _, _, _ = run_kfold(params, patience=10, folds=3)
    return b  # * 0.7 - 0.3 * np.abs(b - c)


def objective_kfold_rac(params):
    K.clear_session()
    _, _, _, b, c, _, _, _, _ = run_kfold(params, key="RAC_regression", patience=10, folds=3)
    return b  # * 0.7 - 0.3 * np.abs(b - c)


######
# NN #
######
os.environ['PYTHONHASHSEED'] = str(random_seed)
keras.utils.set_random_seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
rng = default_rng(random_seed)

make_dir("results/parity_plots_regression")
make_dir("results/learning_curves_regression")


if __name__ == "__main__":
    import gc
    gc.collect()
    result_dict = {}
    # uncomment for hyperparameter training
    # best = fmin(fn=objective_rac_cff, space=space, algo=tpe.suggest, max_evals=100, show_progressbar=True, rstate=rng)
    # best_hp = space_eval(space, best)
    # print("Best hyperparameters:", best_hp)
    best_hp = {'activation': 'leaky_relu', 'batch_size': 32, 'dropout': 0.03, 'hidden_units': (16, 16), 'l2_reg_1': 0.0001991825688908534, 'l2_reg_2': 0.00016125082525844049, 'learning_rate': 0.0016904927957746968}
    history = training(best_hp, key="RAC_cff_regression", return_history=True)
    mae_train, mae_test = history.history["mae"][-1], history.history["val_mae"][-1]
    mse_train, mse_test = history.history["mse"][-1], history.history["val_mse"][-1]
    mape_train, mape_test = history.history["mean_absolute_percentage_error"][-1], history.history["val_mean_absolute_percentage_error"][-1]
    r2 = history.history['r2_score'][-1]
    # TODO: plot learning curves final version
    np.save("results/learning_curves_regression/RAC_cff_train_mse.npy", history.history["mse"])
    np.save("results/learning_curves_regression/RAC_cff_val_mse.npy", history.history["val_mse"])
    np.save("results/learning_curves_regression/RAC_cff_train_mae.npy", history.history["mae"])
    np.save("results/learning_curves_regression/RAC_cff_val_mae.npy", history.history["val_mae"])
    plot_learning_curve_NN(history.history["mse"], history.history["val_mse"], "results/learning_curves_regression/RAC_cff_regression_NN_mse.pdf", True)
    plot_learning_curve_NN(history.history["mae"], history.history["val_mae"], "results/learning_curves_regression/RAC_cff_regression_NN_mae.pdf")
    result_dict["RAC_cff_regression"] = {"mse": mse_test, "mse_train": mse_train, "mae": mae_test, "mae_train": mae_train,
                                         "mape": mape_test, "mape_train": mape_train, "r2": r2, "best_hp": best_hp}
    mae_train, mae_test, mse_train, mse_test, r2, mape_train, mape_test, y_predict_n, y_truth_n = run_kfold(best_hp, key="RAC_cff_regression")
    result_dict["RAC_cff_regression_kfold"] = {"mse": mse_test, "mse_train": mse_train, "mae": mae_test, "mae_train": mae_train,
                                               "mape": mape_test, "mape_train": mape_train, "r2": r2, "best_hp": best_hp}
    plot_parity_plot(y_predict_n=y_predict_n, y_truth_n=y_truth_n, 
                     y_label=r"predicted $\Delta E_\mathsf{HS-LS}$ / kcal mol$^{-1}$",
                     x_label=r"calculated $\Delta E_\mathsf{HS-LS}$ / kcal mol$^{-1}$",
                     filename="results/parity_plots_regression/parity_rac_cff_regression_NN_kfold.pdf")
    print("IDs of Outliers:")
    top_10_outliers = np.argsort(np.abs(np.array(y_predict_n).reshape(-1) - np.array(y_truth_n).reshape(-1)))[::-1][:10]
    print(top_10_outliers)
    print("Predicted Values of Outliers:")
    print(np.array(y_predict_n).reshape(-1)[top_10_outliers])
    print(np.array(y_truth_n).reshape(-1)[top_10_outliers])
    print(result_dict)
    # uncomment for hyperparameter training
    # best = fmin(fn=objective_rac, space=space, algo=tpe.suggest, max_evals=100, show_progressbar=True, rstate=rng)
    # best_hp = space_eval(space, best)
    # print("Best hyperparameters:", best_hp)
    best_hp = {'activation': 'leaky_relu', 'batch_size': 32, 'dropout': 0.19, 'hidden_units': (64, 64), 'l2_reg_1': 0.0005778946971508048, 'l2_reg_2': 0.0002453683368124187, 'learning_rate': 0.001563862198645924}
    history = training(best_hp, key="RAC_regression", return_history=True)
    mae_train, mae_test = history.history["mae"][-1], history.history["val_mae"][-1]
    mse_train, mse_test = history.history["mse"][-1], history.history["val_mse"][-1]
    mape_train, mape_test = history.history["mean_absolute_percentage_error"][-1], history.history["val_mean_absolute_percentage_error"][-1]
    r2 = history.history['r2_score'][-1]
    np.save("results/learning_curves_regression/RAC_train_mse.npy", history.history["mse"])
    np.save("results/learning_curves_regression/RAC_val_mse.npy", history.history["val_mse"])
    np.save("results/learning_curves_regression/RAC_train_mae.npy", history.history["mae"])
    np.save("results/learning_curves_regression/RAC_val_mae.npy", history.history["val_mae"])
    plot_learning_curve_NN(history.history["mse"], history.history["val_mse"], "results/learning_curves_regression/RAC_regression_NN_mse.pdf", True)
    plot_learning_curve_NN(history.history["mae"], history.history["val_mae"], "results/learning_curves_regression/RAC_regression_NN_mae.pdf")
    result_dict["RAC_regression"] = {"mse": mse_test, "mse_train": mse_train, "mae": mae_test, "mae_train": mae_train,
                                     "mape": mape_test, "mape_train": mape_train, "r2": r2, "best_hp": best_hp}
    mae_train, mae_test, mse_train, mse_test, r2, mape_train, mape_test, y_predict_n, y_truth_n = run_kfold(best_hp, key="RAC_regression")
    result_dict["RAC_regression_kfold"] = {"mse": mse_test, "mse_train": mse_train, "mae": mae_test, "mae_train": mae_train,
                                           "mape": mape_test, "mape_train": mape_train, "r2": r2, "best_hp": best_hp}
    plot_parity_plot(y_predict_n=y_predict_n, y_truth_n=y_truth_n,
                     y_label=r"predicted $\Delta E_\mathsf{HS-LS}$ / kcal mol$^{-1}$",
                     x_label=r"calculated $\Delta E_\mathsf{HS-LS}$ / kcal mol$^{-1}$",
                     filename="results/parity_plots_regression/parity_rac_regression_kfold.pdf")
    print("IDs of Outliers:")
    top_10_outliers = np.argsort(np.abs(np.array(y_predict_n).reshape(-1) - np.array(y_truth_n).reshape(-1)))[::-1][:10]
    print(top_10_outliers)
    print("Predicted Values of Outliers:")
    print(np.array(y_predict_n).reshape(-1)[top_10_outliers])
    print(np.array(y_truth_n).reshape(-1)[top_10_outliers])

    print(result_dict)

print("Feature Set, MAPE, MAPE_train, MAE, MAE_train, R2, MSE, MSE_train")
for key in result_dict:
    print(" ".join(key.split("_")), ' ,',
          np.round(result_dict[key]['mape'], 3), ' ,',
          np.round(result_dict[key]['mape_train'], 3), ' ,',
          np.round(result_dict[key]['mae'], 3), ' ,',
          np.round(result_dict[key]['mae_train'], 3), ' ,',
          np.round(result_dict[key]['r2'], 3), ' ,',
          np.round(np.sqrt(result_dict[key]['mse']), 3), ' ,',
          np.round(np.sqrt(result_dict[key]['mse_train']), 3), ' ,')
    
print("")
for key in result_dict:
    print(" ".join(key.split("_")), ' ,',
          np.round(result_dict[key]['mae_train'], 3), ' &',
          np.round(np.sqrt(result_dict[key]['mse_train']), 3), ' &',
          np.round(result_dict[key]['mae'], 3), ' &',
          np.round(np.sqrt(result_dict[key]['mse']), 3), ' &',
          np.round(result_dict[key]['r2'], 3)
          )
