import os
import pandas as pd
import numpy as np
import deepdive as dd
from datetime import datetime
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import matplotlib
import matplotlib.pyplot as plt
now = datetime.now().strftime('%Y%m%d')

config = configparser.ConfigParser()

data_wd = "data"
config_f = "carnivora_pres_div.ini"
config.read(os.path.join(data_wd, config_f))
config.sections()  # see which blocks are listed in the config


config["general"]["wd"] = data_wd
config["empirical_predictions"]["empirical_input_file"] = os.path.join(data_wd, "deepdive_input.csv")
# edit a setting in python
config["simulations"]["n_CPUS"] = '60'
# config["simulations"]["n_training_simulations"] = '10'
# config["simulations"]["n_test_simulations"] = '3'




# Test the simulation settings
# bd_sim, fossil_sim = dd.create_sim_obj_from_config(config)
# sp_x = bd_sim.run_simulation(print_res=True)
# sim = fossil_sim.run_simulation(sp_x)


# Run simulations in parallel
if "simulations" in config.sections():
    feature_file, label_file = dd.run_sim_from_config(config)
else:
    feature_file = None
    label_file = None
if "simulations" in config.sections() and config.getint("simulations", "n_test_simulations"):  # check this functions
    test_feature_file, test_label_file = dd.run_test_sim_from_config(config)
else:
    test_feature_file = None
    test_label_file = None

# Train a model
if "model_training" in config.sections():
    dd.run_model_training_from_config(config, feature_file=feature_file, label_file=label_file)

# run test set
if "n_test_simulations" in list(config["simulations"]):
    pred_list, labels = dd.predict_testset_from_config(config, test_feature_file, test_label_file)
    test_pred = np.squeeze(pred_list[0])
    np.save(os.path.join(config["general"]["wd"], "testset_predictions.npy"), test_pred)
    print("Saved testset predicitons in:\n",
          os.path.join(config["general"]["wd"], "testset_predictions.npy"))

# Predict diversity curves
if "empirical_predictions" in config.sections():
    time_bins = np.sort(list(map(float, config["general"]["time_bins"].split())))

    results = dd.predict_from_config(config)
    pred_div = np.exp(np.squeeze(results[0])) - 1
    pred_div = np.hstack((pred_div[:, 0].reshape(pred_div.shape[0], 1), pred_div))
    pred = np.mean(pred_div, axis=0)

    fig = plt.figure(figsize=(12, 8))
    plt.step(-time_bins, pred.T)

    plt.step(-time_bins,  # pred,
             pred_div.T,
             label="Mean prediction",
             linewidth=2,
             c="b",
             alpha=0.05)

    dd.add_geochrono(0, -4.8, max_ma=-(np.max(time_bins) * 1.05), min_ma=0)
    plt.ylim(bottom=-5, top=np.max(pred) * 1.05)
    plt.xlim(-(np.max(time_bins) * 1.05), -np.min(time_bins) + 2)
    plt.ylabel("Diversity", fontsize=15)
    plt.xlabel("Time (Ma)", fontsize=15)
    file_name = os.path.join(config["general"]["wd"], "predictions.pdf")
    div_plot = matplotlib.backends.backend_pdf.PdfPages(file_name)
    div_plot.savefig(fig)
    div_plot.close()
    print("Plot saved as:", file_name)

    predictions = pd.DataFrame(pred_div)
    predictions.columns = time_bins
    predictions.to_csv(config["empirical_predictions"]['empirical_input_file'] + "_predictions.csv", index=False)
