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
                          pr_extant_clade=float(config["simulations"]["pr_extant_clade"]),
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

    if 2 > 1: #try:
        freq_bin_sampling = config.getfloat("simulations", "bin_sampling")
        bin_mean_rates = np.array(list(map(float, config["simulations"]["bin_mean_rates"].split())))
        bin_std_rates = np.array(list(map(float, config["simulations"]["bin_std_rates"].split())))
    # except:
    #     freq_bin_sampling = 0
    #     bin_std_rates = None
    #     bin_mean_rates = None

    try:
        singleton = config.getfloat("simulations", "singletons_frequency")
    except:
        singleton = list(map(float, config["simulations"]["singletons_frequency"].split()))

    try:
        locality_rate_multiplier = list(map(float, config["simulations"]["locality_rate_multiplier"].split()))
    except:
        locality_rate_multiplier = None

    time_bins = np.sort(np.array(list(map(float, config["general"]["time_bins"].split()))))
    fossil_sim = fossil_simulator(n_areas=config.getint("general", "n_areas"),
                                  n_bins=len(list(map(float, config["general"]["time_bins"].split())))-1,  # number of time bins
                                  time_bins=time_bins,
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
                                  bin_sampling=freq_bin_sampling,
                                  bin_std_rates=bin_std_rates,
                                  bin_mean_rates=bin_mean_rates,
                                  maximum_localities_per_bin=config.getint("simulations", "maximum_localities_per_bin"),
                                  singletons_frequency=singleton,
                                  species_per_locality_multiplier=list(map(float, config["simulations"]["species_per_locality_multiplier"].split())),
                                  locality_rate_multiplier=locality_rate_multiplier,
                                  target_n_occs=target_n_occs,
                                  target_n_occs_range=target_n_occs_range,
                                  seed=rseed)  # if > 0 fixes the random seed to make simulations reproducible
    return bd_sim, fossil_sim


def run_sim_from_config(config):
    # simulate training data
    bd_sim, fossil_sim = create_sim_obj_from_config(config, rseed=config.getint("simulations", "training_seed"))

    try:
        _ = int(config['general']['present_diversity'])
        include_present_diversity = True
    except:
        include_present_diversity = False

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

    n_simulations = config.getint("simulations", "n_training_simulations")
    n_cpus = config.getint("simulations", "n_CPUS")
    if n_cpus > 1:
        n_simulations = int(np.ceil(n_simulations / n_cpus))

    training_set = sim_settings_obj(bd_sim, fossil_sim, n_simulations=n_simulations,
                                    min_age=np.min(list(map(float, config["general"]["time_bins"].split()))),
                                    max_age=np.max(list(map(float, config["general"]["time_bins"].split()))),
                                    seed=config.getint("simulations", "training_seed"), keys=[],
                                    include_present_diversity=include_present_diversity,
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

    try:
        _ = int(config['general']['present_diversity'])
        include_present_diversity = True
    except:
        include_present_diversity = False

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
                                include_present_diversity=include_present_diversity,
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
                    lstm_name = "_".join([str(i) for i in l])
                    dense_name = "_".join([str(i) for i in d])
                    out = 'lstm%s_d%s_%s' % (lstm_name, dense_name, model_tag)
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

def run_model_training_from_config(config, feature_file=None, label_file=None,
                                   convert_to_tf=True, model_tag=None, return_model_dir=False,
                                   calibrate_output=False):
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

    try:
        _ = int(config['general']['present_diversity'])
        include_present_div = True
    except:
        include_present_div = False

        # if config.get("general", "include_present_diversity") == 'TRUE':
        #     include_present_div=True
        #     if config.get("general", "calibrate_diversity") == "TRUE":
        #         calibrate_output = True
        # else:
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

    model_dir = os.path.join(model_wd, out_name)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
    plot_training_history(history, criterion='val_loss', wd=model_dir, show=False, filename=out_name)

    save_rnn_model(model_dir, history, model, feature_rescaler, filename=out_name)

    if return_model_dir:
        return model_dir



def predict_from_config(config, return_features=False,
                        model_tag="", model_dir_id="rnn_model", calibrated=False,
                        return_transformed_diversity=False, model_dir=None):
    dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
    if model_dir is not None:
        loaded_models = load_models(model_wd=model_dir)
    else:
        loaded_models = load_models(model_wd=os.path.join(config["general"]["wd"],
                                                          config["empirical_predictions"]["model_folder"]),
                                    model_name_tag=model_tag, model_dir_id=model_dir_id)

    pres_div = config["general"]["present_diversity"]
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
            pred_list.append(pred_div)
        except:
            pass


    if return_transformed_diversity:
        pred_div = np.exp(np.squeeze(np.array(pred_list))) - 1
        pred_list = np.hstack((pred_div[:, 0].reshape(pred_div.shape[0], 1), pred_div))

    if return_features:
        return pred_list, features
    else:
        return pred_list

def predict_testset_from_config(config, test_feature_file, test_label_file,
                                model_tag="", model_dir_id="rnn_model", calibrated=False,
                                return_features=False, model_dir=None):
    if model_dir is not None:
        loaded_models = load_models(model_wd=model_dir)
    else:
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
                           dropout=False, calibrated=calibrated)

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
                              present_diversity=config.getint("general", "present_diversity"))


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
    print("prop_endemics", prop_endemics)
    config["simulations"]["disp_rate_mean"] = "%s %s" % (0.2 * (1 - prop_endemics), 5 * (1 - prop_endemics))

    indx = np.array([i for i in range(len(feat_names)) if "n_locs_area_" in feat_names[i]])
    # print(feat_names)
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
    config["simulations"]["sd_through_time_skyline"] = "%s" % np.std(np.log(n_localities_area + 1))

    # re-set carrying capacity
    config["simulations"]["dd_K"] = "%s %s" % (np.mean(range_through_div) / 2, np.max(range_through_div) * 5)

    # re-set per-species sampling rate
    m = (n_species - n_singletons)[range_through_div > 0] / range_through_div[range_through_div > 0]
    config["simulations"]["sp_mean"] = "%s %s" % (np.min(m) / 2, np.mean(m))

    # re-set prob gap
    config["simulations"]["p_gap"] = "%s %s" % (0, np.sum(n_occs_area == 0) / np.size(n_occs_area))

    # reset freq singletons
    f_singl_m = np.mean(n_singletons[n_species > 0] / n_species[n_species > 0])
    f_singl_M = np.max(n_singletons[n_species > 1] / n_species[n_species > 1])
    config["simulations"]["singletons_frequency"] = "%s %s" % (f_singl_m, f_singl_M)

    pres_species = int(config["general"]["present_diversity"])
    config["simulations"]["extant_sp"] = "%s %s" % (int(pres_species / 2),
                                                    int(pres_species * 10))

    config["simulations"]["total_sp"] = "%s %s" % (int(np.max(n_species) * 2), int(np.sum(n_species) * 20))

    config["simulations"]["target_n_occs"] = "%s" % np.sum(n_occs)
    config["simulations"]["target_n_occs_range"] = "%s" % target_n_occs_range

    mean_bin_rates = np.mean(n_localities_area, axis=1)
    mean_bin_rates[mean_bin_rates == 0] = np.min(mean_bin_rates[mean_bin_rates > 0])
    list_feat = np.log(mean_bin_rates)

    s = ""
    for i in list_feat:
        s = s + "%s " % i
    config["simulations"]["bin_mean_rates"] = s

    var_bin_rates = np.std(np.log(1 + n_localities_area), axis=1)
    var_bin_rates[var_bin_rates == 0] = np.mean(var_bin_rates[var_bin_rates > 0])
    list_feat = var_bin_rates * 2

    s = ""
    for i in list_feat:
        s = s + "%s " % i
    config["simulations"]["bin_std_rates"] = s

    n_localities_area[n_localities_area == 0] = np.nan
    mean_loc_area = np.nanmean(n_localities_area, axis=0)
    mean_loc_area_r = mean_loc_area / np.mean(mean_loc_area)
    # print("np.mean(n_localities_area, axis=1)", mean_loc_area, mean_loc_area_r)
    s = ""
    for i in mean_loc_area_r:
        s = s + "%s " % i
    config["simulations"]["locality_rate_multiplier"] = s

    config["simulations"]["fraction_skyline_sampling"] = "0.75"
    config["simulations"]["bin_sampling"] = "0.67" # 50% of simulations overall with empirical loc rates

    config["general"]["autotune"] = "FALSE"

    return config

