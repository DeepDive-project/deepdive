import os
import numpy as np
from datetime import datetime
import multiprocessing
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import glob, copy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
from datetime import datetime
from .rnn_builder import fit_rnn
from .bd_simulator import bd_simulator
from .fossil_simulator import fossil_simulator
from .utilities import *
from .feature_extraction import *
from .rnn_builder import *
from .plots import plot_training_history
from .simulation_utilities import *

# create simulator object
def create_sim_obj_from_config(config, rseed=None):
    try:
        s_species = config.getint("simulations", "s_species")
    except:
        s_species = list(map(int, config["simulations"]["s_species"].split()))


    if rseed is None:
        rseed = config.getint("simulations", "training_seed")
    bd_sim = bd_simulator(s_species=s_species,  # number of starting species
                          rangeSP=list(map(float, config["simulations"]["total_sp"].split())),  # min/max size data set
                          minEX_SP=config.getint("simulations", "min_extinct_sp"),  # list(map(float, config["simulations"]["minex_sp"].split())),  # minimum number of extinct lineages
                          root_r=list(map(float, config["simulations"]["root_r"].split())),  # range root ages
                          minEXTANT_SP=np.min(list(map(float, config["simulations"]["extant_sp"].split()))), # min number of living species
                          maxEXTANT_SP=np.max(list(map(float, config["simulations"]["extant_sp"].split()))),
                          rangeL=list(map(float, config["simulations"]["rangel"].split())),  # range of birth rates
                          rangeM=list(map(float, config["simulations"]["rangem"].split())),  # range of death rates
                          log_uniform_rates=config.getboolean("simulations", "log_uniform_rates"),
                          p_mass_extinction=float(config["simulations"]["p_mass_extinction"]),  # probability of mass extinction per my
                          p_equilibrium=config.getfloat("simulations", "p_equilibrium"),
                          p_constant_bd=config.getfloat("simulations", "p_constant_bd"),
                          p_mass_speciation=float(config["simulations"]["p_mass_speciation"]),
                          poiL=config.getfloat("simulations", "poil"),  # expected number of birth rate shifts
                          poiM=config.getfloat("simulations", "poim"),  # expected number of death rate shifts
                          seed=rseed,  # if > 0 fixes the random seed to make simulations reproducible
                          scale=config.getfloat("simulations", "scale"),
                          vectorize=config.getboolean("simulations", "vectorize"))

    # create fossil simulator object
    try:
        target_n_occs = config.getfloat("simulations", "target_n_occs")
        target_n_occs_range = config.getfloat("simulations", "target_n_occs_range")
    except:
        target_n_occs = None
        target_n_occs_range = 10

    fossil_sim = fossil_simulator(n_areas=config.getint("general", "n_areas"),
                                  n_bins=len(list(map(float, config["general"]["time_bins"].split())))-1,  # number of time bins
                                  time_bins=np.array(list(map(float, config["general"]["time_bins"].split())))[::-1],
                                  eta=list(map(float, config["simulations"]["eta"].split())),  # area-sp stochasticity
                                  p_gap=list(map(float, config["simulations"]["p_gap"].split())),  # probability of 0 preservation in a time bin
                                  dispersal_rate=config["simulations"]["dispersal_rate"],
                                  max_dist=config.getfloat("simulations", "max_dist"),
                                  disp_rate_mean=list(map(float, config["simulations"]["disp_rate_mean"].split())),
                                  disp_rate_variance=config.getfloat("simulations", "disp_rate_variance"),
                                  area_mean=config.getfloat("simulations", "area_mean"),
                                  area_variance=config.getfloat("simulations", "area_variance"),
                                  size_concentration_parameter=list(map(float, config["simulations"]["size_concentration_parameter"].split())),
                                  # single value or array of length n_areas
                                  link_area_size_carrying_capacity=list(map(float, config["simulations"]["link_area_size_carrying_capacity"].split())),
                                  # positive, larger numbers = stronger link between area size and carrying capacity
                                  p_origination_a_slope_mean=config.getfloat("simulations", "p_origination_a_slope_mean"),
                                  # mean slope of probability of origination area mean
                                  p_origination_a_slope_sd=config.getfloat("simulations", "p_origination_a_slope_sd"),  # std of the slopes
                                  sp_mean=list(map(float, config["simulations"]["sp_mean"].split())),  # G(a,b) distributed preservation rates across species
                                  sp_variance=config.getfloat("simulations", "sp_variance"),
                                  slope=list(map(float, config["simulations"]["slope"].split())),  # change in log-sampling rate through time (log-linear)
                                  intercept=list(map(float, config["simulations"]["intercept"].split())),  # initial sampling rate
                                  sd_through_time=list(map(float, config["simulations"]["sd_through_time"].split())),
                                  # st dev in log-sampling rate through time
                                  sd_through_time_skyline=config.getfloat("simulations", "sd_through_time_skyline"),
                                  mean_rate_skyline=list(map(float, config["simulations"]["mean_skyline_sampling"].split())),
                                  mean_n_epochs_skyline=config.getfloat("simulations", "mean_n_epochs_skyline"),
                                  fraction_skyline_sampling=config.getfloat("simulations", "fraction_skyline_sampling"),
                                  maximum_localities_per_bin=config.getint("simulations", "maximum_localities_per_bin"),
                                  singletons_frequency=config.getfloat("simulations", "singletons_frequency"),
                                  species_per_locality_multiplier=list(map(float, config["simulations"]["species_per_locality_multiplier"].split())),
                                  target_n_occs=target_n_occs,
                                  target_n_occs_range=target_n_occs_range,
                                  seed=rseed)  # if > 0 fixes the random seed to make simulations reproducible
    return bd_sim, fossil_sim


def run_sim_from_config(config):
    # simulate training data
    bd_sim, fossil_sim = create_sim_obj_from_config(config, rseed=config.getint("simulations", "training_seed"))

    # AREA CONSTRAINTS
    area_start = [i for i in config['simulations'] if 'area_start' in i]
    area_end = [i for i in config['simulations'] if 'area_end' in i]

    # print("area_start", area_start, area_end)
    area_tbl = None
    area_tbl_max = None
    n_areas = int(config['general']['n_areas'])
    if len(area_start):
        area_tbl = np.ones((n_areas, 3))
        area_tbl[:, 0] = np.arange(n_areas)  # set areas IDs
        area_tbl[:, 1:] = -1  # set all values to -1 (all areas exist throughout)
        area_tbl_max = area_tbl + 0

        for i in range(len(area_start)):
            val = list(map(float, config["simulations"][area_start[i]].split()))
            min_val = np.min(val)
            max_val = np.max(val)
            area_tbl[i, 1] = min_val
            area_tbl_max[i, 1] = max_val


    if len(area_end):
        if area_tbl is None:
            area_tbl = np.ones((n_areas, 3))
            area_tbl[:, 0] = np.arange(n_areas)  # set areas IDs
            area_tbl[:, 1:] = -1  # set all values to -1 (all areas exist throughout)
            area_tbl_max = area_tbl + 0

        for i in range(len(area_end)):
            val = list(map(float, config["simulations"][area_end[i]].split()))
            min_val = np.min(val)
            max_val = np.max(val)
            area_tbl[i, 2] = min_val
            area_tbl_max[i, 2] = max_val

    if area_tbl is not None:
        area_constraint = {
            'min_age': area_tbl,
            'max_age': area_tbl_max
        }
    else:
        area_constraint = None

    training_set = sim_settings_obj(bd_sim, fossil_sim, n_simulations=config.getint("simulations", "n_training_simulations"),
                                    min_age=np.min(list(map(float, config["general"]["time_bins"].split()))),
                                    max_age=np.max(list(map(float, config["general"]["time_bins"].split()))),
                                    seed=config.getint("simulations", "training_seed"), keys=[],
                                    include_present_diversity=config.get("general", "include_present_diversity"),
                                    area_constraint=area_constraint
                                    )

    res = run_sim_parallel(training_set, n_CPUS=config.getint("simulations", "n_CPUS"))
    now = datetime.now().strftime('%Y%m%d')
    try:
        os.mkdir(os.path.join(config["general"]["wd"], config["simulations"]["sims_folder"]))
    except FileExistsError:
        pass
    f, l = save_simulations(res, os.path.join(config["general"]["wd"], config["simulations"]["sims_folder"]),
                            config["simulations"]["sim_name"] + "_" + now + "_training", return_file_names=True)
    return f, l


def run_test_sim_from_config(config):
    # simulate test data
    bd_sim, fossil_sim = create_sim_obj_from_config(config, rseed=config.getint("simulations", "test_seed"))

    # AREA CONSTRAINTS
    area_start = [i for i in config['simulations'] if 'area_start' in i]
    area_end = [i for i in config['simulations'] if 'area_end' in i]

    # print("area_start", area_start, area_end)
    area_tbl = None
    area_tbl_max = None
    n_areas = int(config['general']['n_areas'])
    if len(area_start):
        area_tbl = np.ones((n_areas, 3))
        area_tbl[:, 0] = np.arange(n_areas)  # set areas IDs
        area_tbl[:, 1:] = -1  # set all values to -1 (all areas exist throughout)
        area_tbl_max = area_tbl + 0

        for i in range(len(area_start)):
            val = list(map(float, config["simulations"][area_start[i]].split()))
            min_val = np.min(val)
            max_val = np.max(val)
            area_tbl[i, 1] = min_val
            area_tbl_max[i, 1] = max_val


    if len(area_end):
        if area_tbl is None:
            area_tbl = np.ones((n_areas, 3))
            area_tbl[:, 0] = np.arange(n_areas)  # set areas IDs
            area_tbl[:, 1:] = -1  # set all values to -1 (all areas exist throughout)
            area_tbl_max = area_tbl + 0

        for i in range(len(area_end)):
            val = list(map(float, config["simulations"][area_end[i]].split()))
            min_val = np.min(val)
            max_val = np.max(val)
            area_tbl[i, 2] = min_val
            area_tbl_max[i, 2] = max_val

    if area_tbl is not None:
        area_constraint = {
            'min_age': area_tbl,
            'max_age': area_tbl_max
        }
    else:
        area_constraint = None


    test_set = sim_settings_obj(bd_sim, fossil_sim, n_simulations=config.getint("simulations", "n_test_simulations"),
                                min_age=np.min(list(map(float, config["general"]["time_bins"].split()))),
                                max_age=np.max(list(map(float, config["general"]["time_bins"].split()))),
                                seed=config.getint("simulations", "test_seed"),
                                keys=['time_specific_rate', 'species_specific_rate', 'area_specific_rate',
                                'a_var', 'n_bins', 'area_size', 'n_areas', 'n_species', 'n_sampled_species',
                                'tot_br_length', 'n_occurrences', 'slope_pr', 'pr_at_origination',
                                'time_bins_duration', 'eta', 'p_gap', 'area_size_concentration_prm',
                                'link_area_size_carrying_capacity', 'slope_log_sampling',
                                'intercept_initial_sampling', 'sd_through_time', 'additional_info'],
                                include_present_diversity=config.get("general", "include_present_diversity"),
                                area_constraint=area_constraint
                                )

    res = run_sim_parallel(test_set, n_CPUS=config.getint("simulations", "n_CPUS"))
    now = datetime.now().strftime('%Y%m%d')
    f, l = save_simulations(res, os.path.join(config["general"]["wd"], config["simulations"]["sims_folder"]),
                            config["simulations"]["sim_name"] + "_" + now + "_test", return_file_names=True)
    return f, l


def get_model_settings_from_config(config):
    ## Otherwise, how to structure the config that it reads correctly in python? Can't use list
    lstm_nodes = np.array(list(map(int, config["model_training"]["lstm_layers"].split())))
    arrays = [[lstm_nodes[0]]]
    indx = 0
    for i in range(1, len(lstm_nodes)):
        if lstm_nodes[i] < lstm_nodes[i-1]:
            arrays[indx].append(lstm_nodes[i])
        else:
            arrays.append([lstm_nodes[i]])
            indx = indx+1
    print(arrays)
    lstm_nodes = arrays

    dense_nodes = np.array(list(map(int, config["model_training"]["dense_layer"].split())))
    arrays = [[dense_nodes[0]]]
    indx = 0
    for i in range(1, len(dense_nodes)):
        if dense_nodes[i] < dense_nodes[i-1]:
            arrays[indx].append(dense_nodes[i])
        else:
            arrays.append([dense_nodes[i]])
            indx = indx+1
    print(arrays)
    dense_nodes = arrays

    dropout_frac = list(map(float, config["model_training"]["dropout"].split()))
    loss_f = ['mse']

    list_settings = []
    model_n = 0
    try:
        if config["model_training"]["model_tag"] == "NA":
            model_tag = ""
        else:
            model_tag = config["model_training"]["model_tag"]
    except KeyError:
        model_tag = ""
    for l in lstm_nodes:
        for d in dense_nodes:
            for f in loss_f:
                for o in dropout_frac:
                    out = 'lstm%s_d%s_o%s_%s_%s' % (len(l), len(d), o, f, model_tag)
                    d_item = {
                        'model_n': model_n,
                        'lstm_nodes': l,
                        'dense_nodes': d,
                        'loss_f': f,
                        'dropout': o,
                        'model_name': out
                    }
                    list_settings.append(d_item)
                    model_n += 1

    return list_settings

def run_model_training_from_config(config, feature_file=None, label_file=None, convert_to_tf=True, model_tag=None):
    model_settings = get_model_settings_from_config(config)
    sims_path = os.path.join(config["general"]["wd"], config["model_training"]["sims_folder"])
    if feature_file is None:
        feature_file = config["model_training"]["f"]
        label_file = config["model_training"]["l"]
        if config["model_training"]["f"] == "NULL":
            sys.exit("No feature or label files specified, provide to run_model_training or in the config (see R)")
    model_wd = os.path.join(config["general"]["wd"], config["model_training"]["model_folder"]).replace("\\", "/")
    Xt = np.load(os.path.join(sims_path, feature_file))
    Yt = np.load(os.path.join(sims_path, label_file))
    infile_name = os.path.basename(feature_file).split('.npy')[0]
    out_name = infile_name + "_" + model_settings[0]['model_name'] + model_tag

    # feature_rescaler() is a function to rescale the features the same way as done in the training set
    calibrate_output = False
    if config.get("general", "include_present_diversity") == 'TRUE':
        include_present_div=True
        if config.get("general", "calibrate_diversity") == "TRUE":
            calibrate_output = True
    else:
        include_present_div=False
        # remove present diversity if it was included
        Xt = Xt[:, :, 0:len(get_features_names(n_areas=config.getint("general", "n_areas")))]

    # Xt_r, feature_rescaler = normalize_features(Xt, log_last=log_last)
    feature_rescaler = FeatureRescaler(Xt, log_last=include_present_div)
    Xt_r = feature_rescaler.feature_rescale(Xt)

    Yt_r = normalize_labels(Yt, rescaler=1, log=True)

    if convert_to_tf:
        # convert to tf tensors
        Xt_r = np_to_tf(Xt_r)
        Yt_r = np_to_tf(Yt_r)

    if calibrate_output:
        model_config = rnn_config(n_features=Xt_r.shape[2], n_bins=Xt_r.shape[1],
                                  lstm_nodes=model_settings[0]['lstm_nodes'],
                                  dense_nodes=model_settings[0]['dense_nodes'],
                                  calibrate_curve=True,
                                  layers_normalization=False)
        model = build_rnn_model(model_config, print_summary=True)

        present_div_vec = np.einsum('i, ib -> ib', Xt_r[:,0,-1] , np.ones((Xt_r.shape[0], Xt_r.shape[1])))
        dict_inputs = {
            "input_tbl": np_to_tf(Xt_r),
            "present_div": np_to_tf(present_div_vec)
        }

        verbose = 0
        if model_settings[0]['model_n'] == 0:
            verbose = 1
        history = fit_rnn(dict_inputs, Yt_r, model, verbose=verbose,
                          max_epochs=config.getint("model_training", "max_epochs"),
                          patience=config.getint("model_training", "patience"),
                          batch_size=config.getint("model_training", "batch_size"),
                          validation_split=config.getfloat("model_training", "validation_split"))


    else:
        model = build_rnn(Xt_r,
                          lstm_nodes=model_settings[0]['lstm_nodes'],
                          dense_nodes=model_settings[0]['dense_nodes'],
                          loss_f=model_settings[0]['loss_f'],
                          dropout_rate=model_settings[0]['dropout'])

        verbose = 0
        if model_settings[0]['model_n'] == 0:
            verbose = 1
        history = fit_rnn(Xt_r, Yt_r, model, verbose=verbose,
                          max_epochs=config.getint("model_training", "max_epochs"),
                          patience=config.getint("model_training", "patience"),
                          batch_size=config.getint("model_training", "batch_size"),
                          validation_split=config.getfloat("model_training", "validation_split"))
    try:
        os.mkdir(model_wd)
    except FileExistsError:
        pass

    save_rnn_model(model_wd, history, model, feature_rescaler, filename=out_name)
    plot_training_history(history, criterion='val_loss', wd=model_wd, show=False, filename=out_name)


# NEED AN ALTERNATIVE PATH TO PREDICT FROM A TEST SET! SHOULD THIS BE A DIFFERENT FUNCTION OR A DIFFERENT PATH WITHIN
# THIS ONE. CHECK IF THERE'S ANY REDUNDANCY, REMOVE AND RENAME TO MAKE A PLOTTING FUNCTION.
def plot_predictions(config, pred_list):
    fig = plt.figure(figsize=(12, 8))

    # from recent to old
    plot_time_axis = np.sort(list(map(float, config["general"]["time_bins"].split())))

    # from recent to old
    pred_div = predict(features, model, feature_rescaler, n_predictions=n_predictions, dropout=use_dropout)

    pred = np.mean(np.exp(pred_div) - 1, axis=0)
    if scaling == "1-mean":
        den = np.mean(pred)
    elif scaling == "first-bin":
        den = pred[-1]
    else:
        den = 1
    pred /= den

    plt.step(-plot_time_axis,  # pred,
             [pred[0]] + list(pred),
             label="Mean prediction",
             linewidth=2,
             c=prediction_color,
             alpha=0.05)

    predictions.append(pred)

    predictions = np.array(predictions)

    add_geochrono(0, -4.8, max_ma=-66, min_ma=0)
    plt.ylim(bottom=-4.8, top=80)
    plt.xlim(-66, 0)
    plt.ylabel("Diversity", fontsize=15)
    plt.xlabel("Time (Ma)", fontsize=15)
    fig.show()
    file_name = os.path.join(config["general"]["wd"], "predictions.pdf")
    div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
    div_plot.savefig(fig)
    div_plot.close()
    print("Plot saved as:", file_name)

    # Get stats for model training in a pandas dataframe
    # res = list()
    # for i in model_list:
    #     filename = i.split(sep="rnn_model")[1]
    #     history, model, feature_rescaler = dd.load_rnn_model(model_wd, filename=filename)
    #     val_loss = np.min(history["val_loss"])  # check validation loss
    #     t_loss = history["loss"][np.argmin(history["val_loss"])]  # training loss
    #     epochs = np.argmin(history["val_loss"])  # number of epochs used to train
    #     res.append([filename, val_loss, t_loss, epochs])
    # res = pd.DataFrame(res)

    return predictions, res


def run_test_from_config(abs_path,
                         model_wd,
                         new_model_wd,
                         Ytest,
                         Xtest,
                         sqs,
                         output_names,
                         new_output_names=None,
                         test_folder="test"):
    #  predictions for test sets, get stats
    outname = 'sim_features' + output_names[0] + 'lstm3_d2_o0.05_mse'
    print("Running:", outname)
    history, model, feature_rescaler = load_rnn_model(model_wd, filename=outname)
    Ytest_r = normalize_labels(Ytest, rescaler=1, log=True)
    Ytest_pred = predict(features=Xtest, model=model, feature_rescaler=feature_rescaler, n_predictions=10)
    mean_prediction = np.mean(Ytest_pred, axis=0)

    sqs_log_transform = normalize_labels(sqs, rescaler=1, log=True)
    res = calc_time_series_diff2D(mean_prediction, Ytest_r, sqs_log_transform)
    sqs_res = calc_time_series_diff2D(sqs_log_transform, Ytest_r, sqs_log_transform)
    res.to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/' + output_names[0] + '_t_series_diff_2d.csv'),
               index=False)
    sqs_res.to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/' + output_names[0] + '_t_series_diff_2d_sqs.csv'),
                   index=False)

    if new_model_wd is not None:
        new_outname = 'sim_features' + new_output_names[0] + 'lstm3_d2_o0.05_mse'
        print("Running:", new_outname)
        history, model, feature_rescaler = dd.load_rnn_model(new_model_wd, filename=new_outname)
        nYtest_pred = dd.predict(features=Xtest, model=model, feature_rescaler=feature_rescaler, n_predictions=10)
        nmean_prediction = np.mean(nYtest_pred, axis=0)

        res = dd.calc_time_series_diff2D(nmean_prediction, Ytest_r, sqs_log_transform)
        sqs_res = dd.calc_time_series_diff2D(sqs_log_transform, Ytest_r, sqs_log_transform)
        res.to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/' + new_output_names[0] + '_t_series_diff_2d.csv'),
                   index=False)
        sqs_res.to_csv(os.path.join(abs_path + '/test_sets/' + test_folder + '/' + new_output_names[0] + '_t_series_diff_2d_sqs.csv'),
                   index=False)

    return mean_prediction, nmean_prediction, Ytest_r


def predict_from_config(config, return_features=False,
                        model_tag="", model_dir_id="rnn_model", calibrated=False,
                        return_transformed_diversity=False):
    dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
    loaded_models = load_models(model_wd=os.path.join(config["general"]["wd"],
                                                      config["empirical_predictions"]["model_folder"]),
                                model_name_tag=model_tag, model_dir_id=model_dir_id)

    pres_div = config["empirical_predictions"]["present_diversity"]
    if pres_div == "NA":
        features = parse_dd_input(dd_input)
    else:
        features = parse_dd_input(dd_input, present_diversity=int(pres_div))
    pred_list = []
    for model_i in range(len(loaded_models)):
        model = loaded_models[model_i]['model']
        feature_rescaler = loaded_models[model_i]['feature_rescaler']
        try:
            pred_div = predict(features, model, feature_rescaler,
                               n_predictions=config.getint("empirical_predictions", "n_predictions"), dropout=False,
                               calibrated=calibrated)
        except:
            pass

        pred_list.append(pred_div)

    if return_transformed_diversity:
        pred_div = np.exp(np.squeeze(np.array(pred_list))) - 1
        pred_list = np.hstack((pred_div[:, 0].reshape(pred_div.shape[0], 1), pred_div))

    if return_features:
        return pred_list, features
    else:
        return pred_list

def predict_testset_from_config(config, test_feature_file, test_label_file,
                                model_tag="", model_dir_id="rnn_model", calibrated=False,
                                return_features=False):
    loaded_models = load_models(model_wd=os.path.join(config["general"]["wd"],
                                                      config["empirical_predictions"]["model_folder"]),
                                model_name_tag=model_tag, model_dir_id=model_dir_id)


    features = np.load(test_feature_file)
    labels = np.load(test_label_file)
    labels = normalize_labels(labels, rescaler=1, log=True)

    pred_list = []
    for model_i in range(len(loaded_models)):
        model = loaded_models[model_i]['model']
        feature_rescaler = loaded_models[model_i]['feature_rescaler']

        pred_div = predict(features, model, feature_rescaler,
                           n_predictions=config.getint("empirical_predictions", "n_predictions"), dropout=False,
                           calibrated=calibrated)

        pred_list.append(pred_div)

    pred_list = np.squeeze(np.array(pred_list))

    if return_features:
        return pred_list, labels, features
    else:
        return pred_list, labels


def config_autotune(config_init, target_n_occs_range=10):
    config = copy.deepcopy(config_init)
    n_areas = int(config["general"]["n_areas"])
    # load empirical
    dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
    feat_emp = parse_dd_input(dd_input,
                                 present_diversity=config.getint("empirical_predictions", "present_diversity"))


    feat_names_df = get_features_names(n_areas=n_areas, include_present_div=True, as_dataframe=True)
    feat_names = get_features_names(n_areas=n_areas, include_present_div=True, as_dataframe=False)

    time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))
    n_localities = feat_emp[0][:,feat_names.index("n_localities")]
    n_species = feat_emp[0][:,feat_names.index("n_species")]
    n_occs = feat_emp[0][:,feat_names.index("n_occs")]
    range_through_div = feat_emp[0][:, feat_names.index("range_through_div")]
    n_singletons = feat_emp[0][:, feat_names.index("n_singletons")]
    n_endemics = feat_emp[0][:,feat_names.index("n_endemics")]
    prop_endemics = np.mean(n_endemics / (n_species + 1))
    config["simulations"]["disp_rate_mean"] = "%s %s" % (0.5 * (1 / prop_endemics), 2 * (1 / prop_endemics))

    indx = np.array([i for i in range(len(feat_names)) if "n_locs_area_" in feat_names[i]])
    n_localities_area = feat_emp[0][:,indx]

    indx = np.array([i for i in range(len(feat_names)) if "n_species_area_" in feat_names[i]])
    n_species_area = feat_emp[0][:,indx]

    indx = np.array([i for i in range(len(feat_names)) if "n_occs_area_" in feat_names[i]])
    n_occs_area = feat_emp[0][:,indx]

    approx_sampling_rate = np.mean(range_through_div[n_occs > 0] / n_occs[n_occs > 0])

    slopes, intercepts = [], []
    for i in range(n_areas):
        slope, intercept, _, __, ___ = scipy.stats.linregress(time_bins[1:], np.log(n_localities_area[:, i] + 1))
        slopes.append(slope)
        intercepts.append(intercept)

    # re-set (log) slopes in locality rates
    config["simulations"]["slope"] = "%s %s" % (-np.max(np.abs(slopes)) * 2, 0)
    # re-set intercept in locality rates
    config["simulations"]["intercept"] = "%s %s" % (np.min(np.exp(intercepts)) / 2, np.max(np.exp(intercepts)) * 2)

    # re-set mean n. localities per area
    config["simulations"]["area_mean"] = "%s" % np.mean(n_localities_area)
    config["simulations"]["area_variance"] = "%s" % np.var(n_localities_area)

    # re-set max localities per area
    config["simulations"]["maximum_localities_per_bin"] = "%s" % int(np.max(n_localities_area))

    # re-set mean skyline sampling
    config["simulations"]["mean_skyline_sampling"] = "%s %s" % (np.log(np.mean(n_localities_area) / 2),
                                                                np.log(np.mean(n_localities_area) * 2))
    config["simulations"]["sd_through_time_skyline"] = "%s" % np.log(np.var(n_localities_area))

    # re-set carrying capacity
    config["simulations"]["dd_K"] = "%s %s" % (np.mean(range_through_div) / 2, np.max(range_through_div) * 5)

    # re-set per-species sampling rate
    m = (n_species - n_singletons)[range_through_div > 0] / range_through_div[range_through_div > 0]
    config["simulations"]["sp_mean"] = "%s %s" % (np.min(m) / 2, np.mean(m))

    # re-set prob gap
    config["simulations"]["p_gap"] = "%s %s" % (0, np.sum(n_occs_area == 0) / np.size(n_occs_area))

    # reset freq singletons
    f_singl = np.mean(n_singletons[n_species > 0] / n_species[n_species > 0])
    config["simulations"]["p_gap"] = "%s %s" % (f_singl / 2, f_singl * 2)

    pres_species = int(config["empirical_predictions"]["present_diversity"])
    config["simulations"]["extant_sp"] = "%s %s" % (int(pres_species / 2),
                                                    int(pres_species * 10))

    config["simulations"]["total_sp"] = "%s %s" % (int(np.max(n_species) * 2), int(np.sum(n_species) * 2))

    config["simulations"]["target_n_occs"] = "%s" % np.sum(n_occs)
    config["simulations"]["target_n_occs_range"] = "%s" % target_n_occs_range

    return config

