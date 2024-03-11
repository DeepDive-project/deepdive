import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
np.set_printoptions(suppress=True, precision=3)
import os, glob
import pickle as pkl
import scipy.stats

def build_rnn(Xt,
              lstm_nodes=None,
              dense_nodes=None,
              dense_act_f='relu',
              output_nodes=1,
              output_act_f='softplus',
              loss_f='mse',
              dropout_rate=0.2,
              verbose=1):
    if lstm_nodes is None:
        lstm_nodes = [64, 32]
    if dense_nodes is None:
        dense_nodes = [32]
    model = keras.Sequential()

    model.add(
        layers.Bidirectional(layers.LSTM(lstm_nodes[0],
                                         return_sequences=True,
                                         activation='tanh',
                                         recurrent_activation='sigmoid'),
                             input_shape=Xt.shape[1:])
    )
    for i in range(1, len(lstm_nodes)):
        model.add(layers.Bidirectional(layers.LSTM(lstm_nodes[i],
                                                   return_sequences=True,
                                                   activation='tanh',
                                                   recurrent_activation='sigmoid')))
    for i in range(len(dense_nodes)):
        model.add(layers.Dense(dense_nodes[i],
                               activation=dense_act_f))

    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(output_nodes,
                           activation=output_act_f))

    if verbose:
        print(model.summary())

    model.compile(loss=loss_f,
                  optimizer="adam",
                  metrics=['mae', 'mse', 'msle', 'mape'])
    return model


def fit_rnn(Xt, Yt, model,
            criterion="val_loss",
            patience=10,
            verbose=1,
            batch_size=100,
            max_epochs=1000,
            validation_split=0.2
            ):
    early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                               patience=patience,
                                               restore_best_weights=True)
    history = model.fit(Xt, Yt,
                        epochs=max_epochs,
                        validation_split=validation_split,
                        verbose=verbose,
                        callbacks=[early_stop],
                        batch_size=batch_size
                        )
    return history


def save_rnn_model(wd, history, model, feature_rescaler, filename=""):
    # save rescaler
    with open(os.path.join(wd, "rnn_rescaler" + filename + ".pkl").replace("\\", "/"), 'wb') as output:  # Overwrites any existing file.
        pkl.dump(feature_rescaler(1), output, pkl.HIGHEST_PROTOCOL)
    # save training history
    with open(os.path.join(wd, "rnn_history" + filename + ".pkl").replace("\\", "/"), 'wb') as output:  # Overwrites any existing file.
        pkl.dump(history.history, output, pkl.HIGHEST_PROTOCOL)
    # save model
    tf.keras.models.save_model(model, os.path.join(wd, 'rnn_model' + filename).replace("\\", "/"))


def load_rnn_model(wd, filename=""):
    try:
        with open(os.path.join(wd, "rnn_history" + filename + ".pkl"), 'rb') as h:
            history = pkl.load(h)
        with open(os.path.join(wd, "rnn_rescaler" + filename + ".pkl"), 'rb') as h:
            den1d = pkl.load(h)
    except:
        import pickle
        with open(os.path.join(wd, "rnn_history" + filename + ".pkl"), 'rb') as h:
            history = pickle.load(h)
        with open(os.path.join(wd, "rnn_rescaler" + filename + ".pkl"), 'rb') as h:
            den1d = pickle.load(h)

    def feature_rescaler(x):
        return x * den1d
    model = tf.keras.models.load_model(os.path.join(wd, 'rnn_model' + filename))
    return history, model, feature_rescaler


def get_r2(x, y):
    _, _, r_value, _, _ = scipy.stats.linregress(x, y)
    return r_value**2


def get_mse(x, y):
    mse = np.mean((x - y)**2)
    return mse


def get_avg_r2(Ytrue, Ypred):
    r2 = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        r2.append(get_r2(x[x > 0], y[x > 0]))
    res = {'mean r2': np.nanmean(r2),   # change to nanmean? check if this works for sqs data.
           'min r2': np.nanmin(r2),
           'max r2': np.nanmax(r2),
           'std r2': np.nanstd(r2)}
    return res


def get_avg_mse(Ytrue, Ypred):
    mse = []
    if len(Ypred.shape) == 3:
        Ypred = Ypred[:, :, 0]

    for i in range(Ytrue.shape[0]):
        x = Ytrue[i]
        y = Ypred[i, :]
        mse.append(get_mse(x[x > 0], y[x > 0]))
    res = {'mean mse': np.nanmean(mse),
           'min mse': np.nanmin(mse),
           'max mse': np.nanmax(mse),
           'std mse': np.nanstd(mse)}
    return res


def load_models(model_wd, model_name_tag="rnn_model"):
    model_list = glob.glob(os.path.join(model_wd, "*%s*" % model_name_tag))
    models = []
    for model_i in model_list:
        filename = model_i.split(sep="rnn_model")[1]
        print("\nLoading model:", filename)
        history, model, feature_rescaler = load_rnn_model(model_wd, filename=filename)
        models.append({
            'model_name' : filename,
            'history' : history,
            'model': model,
            'feature_rescaler': feature_rescaler
        })

    return models




