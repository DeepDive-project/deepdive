import os
from os import system
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import matplotlib
import matplotlib.pyplot as plt
from .deepdiver_utilities import *
from .plots import add_geochrono_no_labels
from .plots import features_through_time, plot_dd_predictions, plot_ensemble_predictions, features_pca
from .utilities import get_r_squared, get_instance_mse

np.set_printoptions(suppress=True, precision=3)

def run_config(config_file, wd=None, CPU=None, trained_model=None,
               train_set=None, test_set=None, lstm=None, dense=None,
               out_tag="", calibrated=False, total_diversity=None,
               rescale_labels=None,
               n_training_sims=None, n_test_sims=None,
               return_file_names=False,
               verbose=1,
max_iter=None
               ):
    config = configparser.ConfigParser()
    config.read(config_file)

    if wd is not None:
        config["general"]["wd"] = wd

    if total_diversity is None:
        try:
            r_tmp = config["model_training"]["predict_total_diversity"]
            if r_tmp == "TRUE":
                total_diversity = True
            else:
                total_diversity = False
        except:
            total_diversity = False

    try:
        if config["general"]["calibrate_diversity"] == "TRUE":
            calibrated = True
    except:
        pass

    try:
        if config["general"]["present_diversity"] == "NA":
            include_present_diversity = False
        else:
            out_tag = "_conditional"
            include_present_diversity = True
            if calibrated:
                out_tag = "_calibrated"
    except:
        include_present_diversity = False

    if total_diversity:
        out_tag = "_totdiv"


    if rescale_labels is None:
        label_rescaler = None
    else:
        label_rescaler = rescale_labels

    if lstm is not None:
        config["model_training"]["lstm_layers"] = " ".join([str(i) for i in lstm])
    if dense is not None:
        config["model_training"]["dense_layer"] = " ".join([str(i) for i in dense])

    if n_training_sims is not None:
        config["simulations"]["n_training_simulations"] = str(n_training_sims)

    if n_test_sims is not None:
        config["simulations"]["n_test_simulations"] = str(n_test_sims)

    # Run simulations in parallel
    feature_file = None
    label_file = None
    totdiv_label_file = None
    auto_tuned_config_file = None
    if "simulations" in config.sections() and train_set is None and test_set is None:
        if CPU is not None:
            config["simulations"]["n_CPUS"] = str(CPU)

        if config["general"]["autotune"] == "TRUE":
            print("Running autotune...")
            config = config_autotune(config, target_n_occs_range=1.2)
            auto_tuned_config_file = config_file.split(".ini")[0] + "_autotuned.ini"
            with open(auto_tuned_config_file, 'w') as configfile:
                config.write(configfile)

        if config["general"]["autotune"] == "TRUE":
            feature_file, label_file, totdiv_label_file = run_sim_from_config(config)
        else:
            # try:
            feature_file, label_file, totdiv_label_file = run_sim_from_config(config)
    #         except(ValueError):
    #             sys.exit("""
    # Error: Could not run config file - some parameters are missing.
    #        Set autotune=TRUE in the config file or replace all NAs
    #        with appropriate parameter values.
    #        """)


        if total_diversity:
            label_file = totdiv_label_file

    if train_set is not None and trained_model is None:
        if "features.npy" in train_set:
            feature_file = train_set
            if total_diversity:
                label_file = train_set.replace("features.npy", "totdiv.npy")
            else:
                label_file = train_set.replace("features.npy", "labels.npy")
        elif "labels.npy" in train_set:
            feature_file = train_set.replace("labels.npy", "features.npy")
            label_file = train_set
        elif "totdiv.npy" in train_set:
            feature_file = train_set.replace("totdiv.npy", "features.npy")
            label_file = train_set
        else:
            sys.exit("No training features or labels files found")

    test_feature_file = None
    test_label_file = None
    if test_set is not None:
        if "features.npy" in test_set:
            test_feature_file = test_set
            if total_diversity:
                test_label_file = test_set.replace("features.npy", "totdiv.npy")
            else:
                test_label_file = test_set.replace("features.npy", "labels.npy")
        elif "labels.npy" in test_set:
            test_feature_file = test_set.replace("labels.npy", "features.npy")
            test_label_file = test_set
        elif "totdiv.npy" in test_set:
            feature_file = test_set.replace("totdiv.npy", "features.npy")
            label_file = test_set
        else:
            sys.exit("No test features or labels files found")
    else:
        # if train_set is not None:
        #     test_feature_file = feature_file
        #     test_label_file = label_file
        if "simulations" in config.sections() and config.getint("simulations", "n_test_simulations"):
            test_feature_file, test_label_file, test_totdiv_label_file = run_test_sim_from_config(config)

        if total_diversity:
            test_label_file = test_totdiv_label_file

    model_dir = None
    if trained_model is None:
        model_dir = []
        n_models = parse_multiple_models(config)

        # Train a model
        for model_i in range(n_models):
            if feature_file is not None and "model_training" in config.sections():
                model_tmp = run_model_training_from_config(config, feature_file=feature_file, label_file=label_file,
                                                           model_tag=out_tag, return_model_dir=True,
                                                           calibrate_output=calibrated,  total_diversity=total_diversity,
                                                           label_rescaler=label_rescaler,
                                                           verbose=verbose,
                                                           model_number=model_i + 1)
                model_dir.append(model_tmp)
    else:
        model_dir = [trained_model]

    # run test set
    if test_feature_file is not None and model_dir is not None:
        instance_acccuracy = []
        for model_i in range(len(model_dir)):
            test_pred, labels, testset_features = predict_testset_from_config(config,
                                                                              test_feature_file,
                                                                              test_label_file,
                                                                              calibrated=calibrated,
                                                                              return_features=True,
                                                                              model_dir=model_dir[model_i],
                                                                              label_rescaler=label_rescaler
                                                                              )
            # print("test_pred", test_pred, test_feature_file, test_label_file)
            print("\nTest set MSE:", np.mean((test_pred - labels) ** 2))
            pred_file = "testset_pred_%s.npy" % out_tag
            np.save(os.path.join(model_dir[model_i], pred_file), test_pred)
            print("Saved testset predictions in:\n",
                  os.path.join(model_dir[model_i], pred_file))

            with open(file=os.path.join(model_dir[model_i],
                                        pred_file.replace('.npy', 'MSA.txt')),
                      mode='w') as f:
                f.write("Model %s Test set MSE: %s" % (model_i, np.mean((test_pred - labels) ** 2)))


            try:
                if total_diversity is False:
                    instance_acccuracy_m = np.array([get_r_squared(test_pred, labels), get_instance_mse(test_pred, labels)]).T
                    instance_acccuracy.append(instance_acccuracy_m)
                else:
                    instance_acccuracy_m = np.array([np.ones(len(labels)), (test_pred - labels) ** 2]).T
                    instance_acccuracy.append(instance_acccuracy_m)
            except:
                instance_acccuracy = None

    else:
        testset_features = None


    # Predict diversity curves
    if "empirical_predictions" in config.sections():
        # parse area names
        dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
        tbl = pd.read_csv(dd_input)
        area_col = tbl.to_numpy()
        area_names = np.unique(area_col[area_col[:, 0] == 1][3:, 2])

        features_names = get_features_names(n_areas=int(config['general']['n_regions']),
                                            include_present_div=include_present_diversity,
                                            area_names=area_names)
        time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))

        for model_i in range(len(model_dir)):
            pred_div, feat = predict_from_config(config,
                                                 return_features=True,
                                                 model_tag=out_tag,
                                                 calibrated=calibrated,
                                                 return_transformed_diversity=True,
                                                 model_dir=model_dir[model_i])

            if testset_features is not None:
                feature_plot_dir = os.path.join(model_dir[model_i], "feature_plots")
                try:
                    os.mkdir(feature_plot_dir)
                except FileExistsError:
                    pass
                if model_i == 0:
                    plot_feature_hists(test_features=testset_features,
                                       empirical_features=feat[0],
                                       show=False,
                                       n_bins=30,
                                       features_names=features_names,
                                       log_occurrences=True,
                                       wd=feature_plot_dir,
                                       output_name="Feature_plot_log" + out_tag)

                    plot_feature_hists(test_features=testset_features,
                                       empirical_features=feat[0],
                                       show=False,
                                       n_bins=30,
                                       features_names=features_names,
                                       log_occurrences=False,
                                       wd=feature_plot_dir,
                                       output_name="Feature_plot_" + out_tag)

                    # print(time_bins[:-1].shape, testset_features.shape,feat[0].shape)


                    features_through_time(features_names=features_names, time_bins=time_bins,
                                          sim_features=testset_features,
                                          empirical_features=feat[0], wd=feature_plot_dir)
                # plot PCA with accuracy for each model
                features_pca(features_names=features_names, sim_features=testset_features,
                             empirical_features=feat[0],
                             wd=feature_plot_dir,
                             instance_acccuracy=instance_acccuracy[model_i])

            # print(feat.shape, pred_div.shape)

            plot_dd_predictions(pred_div, time_bins, wd=model_dir[model_i],
                                out_tag=out_tag, total_diversity=total_diversity)

            mean_features = np.mean(feat, axis=0)
            feat_tbl = pd.DataFrame(mean_features)
            feat_tbl.columns = features_names
            feat_tbl.to_csv(os.path.join(model_dir[model_i], "Empirical_features_%s.csv" % out_tag),
                               index=False)

            if total_diversity:
                predictions = pd.DataFrame(pred_div.T)
                predictions.columns = ["total_diversity"]
            else:
                predictions = pd.DataFrame(pred_div)
                predictions.columns = time_bins
            predictions.to_csv(os.path.join(model_dir[model_i], "Empirical_predictions_%s.csv" % out_tag),
                               index=False)

        print(
            os.path.join(config["general"]["wd"], config["model_training"]["model_folder"]),
            config["empirical_predictions"]["output_file"],
            total_diversity,
            os.path.join(config["general"]["wd"], config["model_training"]["model_folder"])
        )


        if len(model_dir) > 1:
            plot_ensemble_predictions(model_wd=os.path.join(config["general"]["wd"], config["model_training"]["model_folder"]),
                                      out_tag=config["empirical_predictions"]["output_file"],
                                      tot_div=total_diversity,
                                      wd=os.path.join(config["general"]["wd"], config["model_training"]["model_folder"]))

        # save config used for the analysis in the model folder
        with open(os.path.join(config["general"]["wd"], config["model_training"]["model_folder"], "config_settings.txt"), 'w') as configfile:
            config.write(configfile)

        if return_file_names:
            out_files = {
                'training_features': feature_file,
                'test_features': test_feature_file,
                'training_labels': label_file,
                'test_labels': test_label_file,
                'training_tot_div': totdiv_label_file,
                'test_total_div': test_totdiv_label_file,
                'model_dir': model_dir

            }
            return out_files


def sim_and_plot_features(config_file, wd=None, CPU=None, n_sims=None):

    config = configparser.ConfigParser()
    config.read(config_file)

    if wd is not None:
        config["general"]["wd"] = wd


    try:
        pres_div = config["general"]["present_diversity"]
    except KeyError:
        pres_div = "NA"

    if pres_div == "NA":
        include_present_diversity = False
    else:
        include_present_diversity = True

    # Run simulations in parallel
    if CPU is not None:
        config["simulations"]["n_CPUS"] = str(CPU)

    if n_sims is not None:
        config["simulations"]["n_test_simulations"] = str(n_sims)

    if config["general"]["autotune"] == "TRUE":
        print("Running autotune...")
        config = config_autotune(config, target_n_occs_range=1.2)
        auto_tuned_config_file = config_file.split(".ini")[0] + "_autotuned.ini"
        with open(auto_tuned_config_file, 'w') as configfile:
            config.write(configfile)


    test_feature_file, test_label_file, totdiv_label_file = run_test_sim_from_config(config)
    testset_features = np.load(test_feature_file)
    feature_plot_dir = os.path.join(config["general"]["wd"],
                                    config["empirical_predictions"]["output_file"] + "_feature_plots")

    # load empirical features
    # parse area names
    dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
    tbl = pd.read_csv(dd_input)
    area_col = tbl.to_numpy()
    area_names = np.unique(area_col[area_col[:,0] == 1][3:, 2])
    # print("area_names", area_names)

    features_names = get_features_names(n_areas=int(config['general']['n_regions']),
                                        include_present_div=include_present_diversity, area_names=area_names)
    # print("features", features_names)

    time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))

    dd_input = os.path.join(config["general"]["wd"], config["empirical_predictions"]["empirical_input_file"])
    if pres_div == "NA":
        feat = parse_dd_input(dd_input)
    else:
        feat = parse_dd_input(dd_input, present_diversity=int(pres_div))


    try:
        os.mkdir(feature_plot_dir)
    except FileExistsError:
        pass
    plot_feature_hists(test_features=testset_features,
                       empirical_features=feat[0],
                       show=False,
                       n_bins=30,
                       features_names=features_names,
                       log_occurrences=True,
                       wd=feature_plot_dir,
                       output_name="Feature_plot_log")

    plot_feature_hists(test_features=testset_features,
                       empirical_features=feat[0],
                       show=False,
                       n_bins=30,
                       features_names=features_names,
                       log_occurrences=False,
                       wd=feature_plot_dir,
                       output_name="Feature_plot")

    # print(time_bins[:-1].shape, testset_features.shape,feat[0].shape)

    # print("sim_features.shape", testset_features.shape, feat[0].shape)


    features_through_time(features_names=features_names, time_bins=time_bins,
                          sim_features=testset_features,
                          empirical_features=feat[0], wd=feature_plot_dir)

    features_pca(features_names=features_names, sim_features=testset_features,
                 empirical_features=feat[0], wd=feature_plot_dir)

    print("Feature plots saved in {}".format(feature_plot_dir))


def run_autotune(config_file, wd=None):

    config = configparser.ConfigParser()
    config.read(config_file)

    if wd is not None:
        config["general"]["wd"] = wd

    print("Running autotune...")
    config = config_autotune(config, target_n_occs_range=1.2)
    auto_tuned_config_file = config_file.split(".ini")[0] + "_autotuned.ini"
    with open(auto_tuned_config_file, 'w') as configfile:
        config.write(configfile)
