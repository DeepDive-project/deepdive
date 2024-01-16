import os
import deepdive as dd
import numpy as np
from datetime import datetime
import multiprocessing
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import glob


# create simulator object
def create_sim_obj_from_config(config, rseed=None):
    if rseed is None:
        rseed = config.getint("simulations", "training_seed")
    bd_sim = dd.bd_simulator(s_species=config.getint("simulations", "s_species"),  # number of starting species
                             rangeSP=list(map(float, config["simulations"]["rangesp"].split())),  # min/max size data set
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
                             p_mass_speciation=list(map(float, config["simulations"]["p_mass_speciation"].split())),
                             poiL=config.getfloat("simulations", "poil"),  # expected number of birth rate shifts
                             poiM=config.getfloat("simulations", "poim"),  # expected number of death rate shifts
                             seed=rseed,  # if > 0 fixes the random seed to make simulations reproducible
                             scale=config.getfloat("simulations", "scale"),
                             vectorize=config.getboolean("simulations", "vectorize"))

    # create fossil simulator object
    fossil_sim = dd.fossil_simulator(n_areas=config.getint("simulations", "n_areas"),
                                     n_bins=len(list(map(float, config["simulations"]["time_bins"].split()))),  # number of time bins
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
                                     mean_n_epochs_skyline=config.getfloat("simulations", "mean_n_epochs_skyline"),
                                     fraction_skyline_sampling=config.getfloat("simulations", "fraction_skyline_sampling"),
                                     maximum_localities_per_bin=config.getint("simulations", "maximum_localities_per_bin"),
                                     singletons_frequency=list(map(float, config["simulations"]["singletons_frequency"].split())),
                                     seed=rseed)  # if > 0 fixes the random seed to make simulations reproducible
    return bd_sim, fossil_sim


def run_sim(rep, config):
    batch_features = []
    batch_labels = []

    # simulate training data
    bd_sim, fossil_sim = create_sim_obj_from_config(config.getint("simulations", "training_seed") + rep, config=config)
    for i in range(config.getint("simulations", "n_training_simulations")):
        if i % 1 == 0 and rep == 0:
            dd.print_update("%s of %s done" % (i + 1, config.getint("simulations", "n_training_simulations")))
            sp_x = bd_sim.run_simulation(print_res=False)

        # ----
        # if config.getboolean("simulations", "use_area_constraints"):
        #     c1, c2 = dd.set_area_constraints(sp_x=sp_x,
        #                                      n_time_bins=len(list(map(float, config["simulations"]["time_bins"].split()))),
        #                                      area_tbl=area_tbl,
        #                                      mid_time_bins=mid_time_bins)           ### How do we automate this? Where is it being read form in the empirical files???
        #
        #     fossil_sim.set_carrying_capacity_multiplier(m_species_origin=c1, m_sp_area_time=c2)
        # ----

        # using min_age and max_age ensures that the time bins always span the same amount of time
        sim = fossil_sim.run_simulation(sp_x, min_age=0, max_age=config.getfloat("simulations", "max_clade_age"))  # MIN AGE = 0 SHOULDN'T BE HARD CODED

        # ----
        sim_features = dd.extract_sim_features(sim)
        sim_y = sim['global_true_trajectory']
        batch_features.append(sim_features)
        batch_labels.append(sim_y)

    res = {'features': batch_features,
           'labels': batch_labels}
    return res


def run_test_sim(rep, config):
    batch_features = []
    batch_labels = []

    # simulate training data
    sim_settings = []
    bd_sim, fossil_sim = create_sim_obj_from_config(config.getint("simulations", "test_seed") + rep, config=config)
    for i in range(config.getint("simulations", "n_test_simulations")):
        if i % 1 == 0 and rep == 0:
            dd.print_update("%s of %s done" % (i + 1, config.getfloat("simulations", "n_test_simulations")))
        sp_x = bd_sim.run_simulation(print_res=False)
        # ----
        # if config.getboolean("simulations", "use_area_constraints"):
        #     c1, c2 = dd.set_area_constraints(sp_x=sp_x,
        #                                      n_time_bins=len(list(map(float, config["simulations"]["time_bins"].split()))),
        #                                      area_tbl=area_tbl,
        #                                      mid_time_bins=mid_time_bins)
        #
        #     fossil_sim.set_carrying_capacity_multiplier(m_species_origin=c1, m_sp_area_time=c2)
        # ----

        # using min_age and max_age ensures that the time bins always span the same amount of time
        sim = fossil_sim.run_simulation(sp_x, min_age=0, max_age=config.getfloat("simulations", "max_clade_age"))  # MIN AGE = 0 SHOULDN'T BE HARD CODED
        sim_features = dd.extract_sim_features(sim)
        sim_y = sim['global_true_trajectory']
        batch_features.append(sim_features)
        batch_labels.append(sim_y)

        keys = ['time_specific_rate', 'species_specific_rate', 'area_specific_rate',
                'a_var', 'n_bins', 'area_size', 'n_areas', 'n_species', 'n_sampled_species', 'tot_br_length',
                'n_occurrences', 'slope_pr', 'pr_at_origination', 'time_bins_duration', 'eta', 'p_gap',
                'area_size_concentration_prm', 'link_area_size_carrying_capacity',
                'slope_log_sampling', 'intercept_initial_sampling', 'sd_through_time', 'additional_info']

        s = {key: sim[key] for key in keys}
        sim_settings.append(s)

    res = {'features': batch_features,
           'labels': batch_labels,
           'settings': sim_settings}
    return res


def get_model_settings(config):  ## IDEAS TO FIX ERROR: is it better to hard code the nodes and just have the user specify the number of layers for them?
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

    dense_nodes = list(map(int, config["model_training"]["dense_layer"].split()))
    arrays = [[dense_nodes[0]]]
    indx = 0
    for i in range(1, len(dense_nodes)):
        if dense_nodes[i] < dense_nodes[i-1]:
            arrays[indx].append(dense_nodes[i])
        else:
            arrays.append([dense_nodes[i]])
            indx = indx+1
    print(arrays)
    lstm_nodes = arrays

    dropout_frac = list(map(float, config["model_training"]["dropout"].split()))
    loss_f = ['mse']

    list_settings = []
    model_n = 0
    for l in lstm_nodes:
        for d in dense_nodes:
            for f in loss_f:
                for o in dropout_frac:
                    out = 'lstm%s_d%s_o%s_%s' % (len(l), len(d), o, f)
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

def run_model_training(config, wd, d, feat_file, model_wd):
    Xt = np.load(os.path.join(wd, d['feature_file']))
    Yt = np.load(os.path.join(wd, d['label_file']))
    infile_name = feat_file.split('.npy')[0]
    out_name = infile_name + d['model_name']

    # feature_rescaler() is a function to rescale the features the same way as done in the training set
    Xt_r, feature_rescaler = dd.normalize_features(Xt)
    Yt_r = dd.normalize_labels(Yt, rescaler=1, log=True)
    model = dd.build_rnn(Xt_r,
                         lstm_nodes=d['lstm_nodes'],
                         dense_nodes=d['dense_nodes'],
                         loss_f=d['loss_f'],
                         dropout_rate=d['dropout'])
    verbose = 0
    if d['model_n'] == 0:
        verbose = 1
    history = dd.fit_rnn(Xt_r, Yt_r, model, verbose=verbose,
                         max_epochs=config.getint("model_training", "max_epochs"),
                         patience=config.getint("model_training", "patience"),
                         batch_size=config.getint("model_training", "batch_size"),
                         validation_split=config.getfloat("model_training", "validation_split"))
    dd.save_rnn_model(model_wd, history, model, feature_rescaler, filename=out_name)
    dd.plot_training_history(history, criterion='val_loss', wd=model_wd, show=False, filename=out_name)


def predict(config):
    np.random.seed(config.getint("predictions", "random_seed"))

    # Specify settings
    n_predictions = config.getint("predictions", "n_predictions")  # number of predictions per input file
    replicates = config.getint("predictions", "replicates")  # number of age randomisation replicates in data_pipeline.R
    alpha = config.getfloar("predictions", "alpha")
    prediction_color = config["predictions"]["prediction_color"]
    # scaling_options: None, "1-mean", "first-bin"
    scaling = config["predictions"]["scaling"]
    plot_shaded_area = config  # Resolve the errors this will make here.
    combine_all_models = True  # if false plot each model separately  - is this a relict (remove) or something that should be moved to config? should be model ensemble now instead

    # run predictions across all models
    model_list = glob.glob(os.path.join(model_wd, "*rnn_model*"))

    # create time bin indices from recent to old
    time_bins = np.sort(np.array(list(map(float, config["simulations"]["time_bins"].split()))))


def run_test(abs_path,
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
    history, model, feature_rescaler = dd.load_rnn_model(model_wd, filename=outname)
    Ytest_r = dd.normalize_labels(Ytest, rescaler=1, log=True)
    Ytest_pred = dd.predict(features=Xtest, model=model, feature_rescaler=feature_rescaler, n_predictions=10)
    mean_prediction = np.mean(Ytest_pred, axis=0)

    sqs_log_transform = dd.normalize_labels(sqs, rescaler=1, log=True)
    res = dd.calc_time_series_diff2D(mean_prediction, Ytest_r, sqs_log_transform)
    sqs_res = dd.calc_time_series_diff2D(sqs_log_transform, Ytest_r, sqs_log_transform)
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



# Import config function
# def run_config():
#     config = configparser.ConfigParser()
#     config.read("config.ini")
#     config.sections()  # see which blocks are listed in the config
#     # "simulations" in config  # to see if a block is present in the config, returns True/False
#     # config["simulations"]["eta"]  # to call settings
#     #for key in config["simulations"]:  # lists settings included in a block
#     #    print(key)
#
#     if config.getint("simulations", "n_training_simulations"):
#         create_sim_obj_from_config(rseed, config)
#         run_sim(rep, config)
#
#     if config.getint("simulations", "n_training_simulations"):
#         create_sim_obj_from_config(rseed, config)
#         run_test_sim(rep, config)
#
#     if config.getint("model_training"):
#         get_model_settings()
#         run_model_training()
#
#     if :
#         run_test()
#
#     if :
#         predict()
