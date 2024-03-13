import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
np.set_printoptions(suppress=True, precision=3)
import os, glob
import pickle as pkl
import scipy.stats

from keras.layers import Activation
from tensorflow.python.keras.utils import generic_utils

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
                  metrics=['mae']) # , 'msle', 'mape', 'mse'
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

class rnn_config():
    def __init__(self,
                 lstm_nodes: list = None,
                 nn_dense: list = None,
                 pool_per_bin = True,
                 mean_normalize_rates = True,
                 layers_normalization = False,
                 output_f: list = None,
                 n_features: int = None,
                 n_bins: int = None,
                 ):

        if lstm_nodes is None:
            lstm_nodes = [128, 64]
        if nn_dense is None:
            nn_dense = [64, 32]
        if output_f is None:
            output_f = 'softplus'

        self.lstm_nodes = lstm_nodes
        self.nn_dense = nn_dense
        self.mean_normalize_rates = mean_normalize_rates
        self.layers_norm = layers_normalization
        self.output_f = output_f
        self.n_features = n_features
        self.n_bins = n_bins
        self.pool_per_bin = pool_per_bin

def build_rnn_model(model_config: rnn_config,
                    optimizer=keras.optimizers.RMSprop(1e-3),
                    print_summary=False
                    ):
    ali_input = keras.Input(shape=(model_config.n_bins, model_config.n_features,),
                            name="input_tbl")
    print("SHAPE ali_input", ali_input.shape)
    # inputs = [ali_input]

    # lstm on sequence data

    ali_rnn_1 = layers.Bidirectional(
        layers.LSTM(model_config.lstm_nodes[0], return_sequences=True, activation='tanh',
                    recurrent_activation='sigmoid', name="sequence_LSTM_1"))(ali_input)
    if model_config.layers_norm:
        ali_rnn_1n = layers.LayerNormalization(name='layer_norm_rnn1')(ali_rnn_1)
    else:
        ali_rnn_1n = ali_rnn_1
    ali_rnn_2 = layers.Bidirectional(
        layers.LSTM(model_config.lstm_nodes[1], return_sequences=True, activation='tanh',
                    recurrent_activation='sigmoid', name="sequence_LSTM_2"))(ali_rnn_1n)
    if model_config.layers_norm:
        ali_rnn_2n = layers.LayerNormalization(name='layer_norm_rnn2')(ali_rnn_2)
    else:
        ali_rnn_2n = ali_rnn_2


    #--- block w shared prms
    site_dnn_1 = layers.Dense(model_config.nn_dense[0], activation='swish', name="site_NN")

    print("Creating blocks...")
    comb_outputs = [layers.Flatten()(i) for i in tf.split(ali_rnn_2n, model_config.n_bins, axis=1)]

    site_sp_dnn_1_list = [site_dnn_1(i) for i in comb_outputs]

    print("done")
    #---

    # Merge all available features into a single large vector via concatenation
    # concat_1 = layers.concatenate([ali_rnn_3, phy_dnn_1])
    outputs = []
    loss = {}
    loss_w = {}
    # output: diversity per bin
    site_rate_1 = layers.Dense(model_config.nn_dense[1], activation='swish', name="site_rate_hidden")
    if len(model_config.nn_dense) > 2:
        print("Warning: only tywo dense layers are currently supported!")
    site_rate_1_list = [site_rate_1(i) for i in site_sp_dnn_1_list]
    rate_pred_nn = layers.Dense(1, activation=model_config.output_f, name="per_site_rate_split")
    rate_pred_list = [rate_pred_nn(i) for i in site_rate_1_list]
    # rate_pred =  layers.Dense(1, activation='linear', name="per_site_rate")(layers.concatenate(rate_pred_list))

    if not model_config.mean_normalize_rates:
        rate_pred = layers.Flatten(name="per_site_rate")(layers.concatenate(rate_pred_list))
    else:
        def mean_rescale(x):
            return x / x[0]
            # return x / tf.reduce_mean(x, axis=1, keepdims=True)

        generic_utils.get_custom_objects().update({'mean_rescale': Activation(mean_rescale)})
        rate_pred_tmp = layers.Flatten(name="per_site_rate_tmp")(layers.concatenate(rate_pred_list))
        rate_pred = layers.Activation(mean_rescale, name='per_site_rate')(rate_pred_tmp)

    outputs.append(rate_pred)
    loss['per_site_rate'] = keras.losses.MeanSquaredError()
    loss_w["per_site_rate"] = 1



    # Instantiate an end-to-end model predicting both rates and substitution model
    model = keras.Model(
        inputs=ali_input,
        outputs=outputs,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_w
    )

    if print_summary:
        print(model.summary())

    print("N. model parameters:", model.count_params())

    return model



