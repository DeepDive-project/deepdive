import os
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import matplotlib
import matplotlib.pyplot as plt
from .deepdiver_utilities import *
from .plots import add_geochrono

np.set_printoptions(suppress=True, precision=3)

def run_config(config_file, wd=None, CPU=None):
    config = configparser.ConfigParser()
    config.read(config_file)

    if wd is not None:
        config["general"]["wd"] = wd

    if config["general"]["include_present_diversity"] == "FALSE":
        out_tag = "conditional"
    else:
        out_tag = "unconditional"
    if config["general"]["calibrate_diversity"] == "TRUE":
        calibrated = True
    else:
        calibrated = False
        out_tag = "calibrated"

    # Run simulations in parallel
    if "simulations" in config.sections():
        if CPU is not None:
            config["simulations"]["n_CPUS"] = str(CPU)
            # print("CPU", config["simulations"]["n_CPUS"] , CPU)

        if config["general"]["autotune"] == "TRUE":
            print("Running autotune...")
            config = config_autotune(config)
            auto_tuned_config_file = config_file.split(".ini")[0] + "_autotuned.ini"
            with open(auto_tuned_config_file, 'w') as configfile:
                config.write(configfile)

        feature_file, label_file = run_sim_from_config(config)

    if "simulations" in config.sections() and config.getint("simulations", "n_test_simulations"):
        test_feature_file, test_label_file = run_test_sim_from_config(config)

    # Train a model
    if "model_training" in config.sections():
        run_model_training_from_config(config, feature_file=feature_file, label_file=label_file)

    # run test set
    if "simulations" in config.sections() and "model_training" in config.sections():
        test_pred, labels = predict_testset_from_config(config,
                                                        test_feature_file,
                                                        test_label_file,
                                                        model_tag=out_tag,
                                                        calibrated=calibrated)

        print("Test set MSE:", np.mean((test_pred - labels) ** 2))
        pred_file = "testset_pred_%s.npy" % out_tag
        np.save(os.path.join(config["general"]["wd"], pred_file), test_pred)
        print("Saved testset predictions in:\n",
              os.path.join(config["general"]["wd"], pred_file))

    # Predict diversity curves
    if "empirical_predictions" in config.sections():
        time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))

        pred_div, feat = predict_from_config(config,
                                             return_features=True,
                                             model_tag=out_tag,
                                             calibrated=calibrated,
                                             return_transformed_diversity=True)

        pred = np.mean(pred_div, axis=0)

        fig = plt.figure(figsize=(12, 8))
        plt.step(-time_bins, pred.T)

        plt.step(-time_bins,
                 pred_div.T,
                 label="Mean prediction",
                 linewidth=2,
                 c="b",
                 alpha=0.05)

        add_geochrono(0, -4.8, max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
        plt.ylim(bottom=-5, top=np.max(pred) * 1.05)
        plt.xlim(-(np.max(time_bins) * 1.05), -np.min(time_bins) + 2)
        plt.ylabel("Diversity", fontsize=15)
        plt.xlabel("Time (Ma)", fontsize=15)
        file_name = os.path.join(config["general"]["wd"], "predictions_%s.pdf" % out_tag)
        div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
        div_plot.savefig(fig)
        div_plot.close()
        print("Plot saved as:", file_name)

        predictions = pd.DataFrame(pred_div)
        predictions.columns = time_bins
        predictions.to_csv(config["empirical_predictions"]['empirical_input_file'] + "_predictions_%s.csv" % out_tag,
                           index=False)