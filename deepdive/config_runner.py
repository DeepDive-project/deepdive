import os
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import matplotlib
import matplotlib.pyplot as plt
from .deepdiver_utilities import *
from .plots import add_geochrono_no_labels

np.set_printoptions(suppress=True, precision=3)

def run_config(config_file, wd=None, CPU=None, trained_model=None,
               train_set=None, test_set=None, lstm=None, dense=None):
    config = configparser.ConfigParser()
    config.read(config_file)

    if wd is not None:
        config["general"]["wd"] = wd

    if config["general"]["include_present_diversity"] == "FALSE":
        out_tag = "unconditional"
        include_present_diversity = False
    else:
        out_tag = "conditional"
        include_present_diversity = True
    if config["general"]["calibrate_diversity"] == "TRUE":
        calibrated = True
        out_tag = "calibrated"
    else:
        calibrated = False

    if lstm is not None:
        config["model_training"]["lstm_layers"] = " ".join([str(i) for i in lstm])
    if dense is not None:
        config["model_training"]["dense_layer"] = " ".join([str(i) for i in dense])

    # Run simulations in parallel
    feature_file = None
    if "simulations" in config.sections() and train_set is None and trained_model is None:
        if CPU is not None:
            config["simulations"]["n_CPUS"] = str(CPU)
            # print("CPU", config["simulations"]["n_CPUS"] , CPU)

        if config["general"]["autotune"] == "TRUE":
            print("Running autotune...")
            config = config_autotune(config, target_n_occs_range=1.2)
            auto_tuned_config_file = config_file.split(".ini")[0] + "_autotuned.ini"
            with open(auto_tuned_config_file, 'w') as configfile:
                config.write(configfile)

        feature_file, label_file = run_sim_from_config(config)

    if train_set is not None and trained_model is None:
        if "features.npy" in train_set:
            feature_file = train_set
            label_file = train_set.replace("features.npy", "labels.npy")
        elif "labels.npy" in train_set:
            feature_file = train_set.replace("labels.npy", "features.npy")
            label_file = train_set
        else:
            sys.exit("No training features or labels files found")

    test_feature_file = None

    if test_set is not None:
        if "features.npy" in test_set:
            test_feature_file = test_set
            test_label_file = test_set.replace("features.npy", "labels.npy")
        elif "labels.npy" in test_set:
            test_feature_file = test_set.replace("labels.npy", "features.npy")
            test_label_file = test_set
        else:
            sys.exit("No test features or labels files found")
    else:
        if "simulations" in config.sections() and config.getint("simulations", "n_test_simulations"):
            test_feature_file, test_label_file = run_test_sim_from_config(config)

    model_dir = None
    if trained_model is None:
        # Train a model
        if feature_file is not None and "model_training" in config.sections():
            model_dir = run_model_training_from_config(config, feature_file=feature_file, label_file=label_file,
                                           model_tag=out_tag, return_model_dir=True)
    else:
        model_dir = trained_model

    # run test set
    if test_feature_file is not None and model_dir is not None:
        test_pred, labels, testset_features = predict_testset_from_config(config,
                                                                          test_feature_file,
                                                                          test_label_file,
                                                                          calibrated=calibrated,
                                                                          return_features=True,
                                                                          model_dir=model_dir
                                                                          )
        print("test_pred", test_pred, test_feature_file, test_label_file)
        print("Test set MSE:", np.mean((test_pred - labels) ** 2))
        pred_file = "testset_pred_%s.npy" % out_tag
        np.save(os.path.join(model_dir, pred_file), test_pred)
        print("Saved testset predictions in:\n",
              os.path.join(model_dir, pred_file))
    else:
        testset_features = None


    # Predict diversity curves
    if "empirical_predictions" in config.sections():
        features_names = get_features_names(n_areas=int(config['general']['n_areas']),
                                            include_present_div=include_present_diversity)
        time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))

        pred_div, feat = predict_from_config(config,
                                             return_features=True,
                                             model_tag=out_tag,
                                             calibrated=calibrated,
                                             return_transformed_diversity=True,
                                             model_dir=model_dir)

        if testset_features is not None:
            plot_feature_hists(test_features=testset_features,
                               empirical_features=feat[0],
                               show=False,
                               n_bins=30,
                               features_names=features_names,
                               log_occurrences=True,
                               wd=model_dir,
                               output_name="Feature_plot_log" + out_tag)

            plot_feature_hists(test_features=testset_features,
                               empirical_features=feat[0],
                               show=False,
                               n_bins=30,
                               features_names=features_names,
                               log_occurrences=False,
                               wd=model_dir,
                               output_name="Feature_plot_" + out_tag)

        print(feat.shape, pred_div.shape)

        pred = np.mean(pred_div, axis=0)

        fig = plt.figure(figsize=(12, 8))
        plt.step(-time_bins, pred.T)

        plt.step(-time_bins,
                 pred_div.T,
                 label="Mean prediction",
                 linewidth=2,
                 c="b",
                 alpha=0.05)

        rt_div = np.mean(feat[:, :, 5], axis=0)
        rt_div = np.array([rt_div[0]] + list(rt_div)) 
        print(rt_div.shape)
        plt.step(-time_bins, rt_div,
                 label="Range-through",
                 linewidth=2,
                 )

        add_geochrono_no_labels(0, -0.1 * np.max(pred), max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
        plt.ylim(bottom=-0.1*np.max(pred), top=np.max(pred) * 1.05)
        plt.xlim(-(np.max(time_bins) * 1.05), -np.min(time_bins) + 2)
        plt.ylabel("Diversity", fontsize=15)
        plt.xlabel("Time (Ma)", fontsize=15)
        file_name = os.path.join(model_dir, "Empirical_predictions_%s.pdf" % out_tag)
        div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
        div_plot.savefig(fig)
        div_plot.close()
        print("Plot saved as:", file_name)

        predictions = pd.DataFrame(pred_div)
        predictions.columns = time_bins
        predictions.to_csv(os.path.join(model_dir, "Empirical_predictions_%s.csv" % out_tag),
                           index=False)
