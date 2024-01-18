import os
import deepdive as dd
import numpy as np
from datetime import datetime
import multiprocessing
import configparser  # read the config file (".ini") created by the r function 'create_config()'
import glob

now = datetime.now().strftime('%Y%m%d')

config = configparser.ConfigParser()

# wd = "/Users/dsilvestro/Software/DeepDive-project/deepdive/test_deepdiveR/"
wd = "/Users/CooperR/Documents/GitHub/deep_dive/"
config_f = "try1.ini"
config.read(os.path.join(wd, config_f))
config.sections()  # see which blocks are listed in the config
# "simulations" in config  # to see if a block is present in the config, returns True/False
# config["simulations"]["eta"]  # to call settings


bd_sim, fossil_sim = dd.create_sim_obj_from_config(config)


# Test the simulation settings
sp_x = bd_sim.run_simulation(print_res=True)
sim = fossil_sim.run_simulation(sp_x)

# Run simulations in parallel
if config.getint("simulations", "n_training_simulations"):
    training_set = dd.sim_settings_obj(bd_sim,
                                       fossil_sim,
                                       n_simulations=config.getint("simulations", "n_training_simulations"),
                                       min_age=np.min(list(map(float, config["simulations"]["time_bins"].split()))),
                                       max_age=np.max(list(map(float, config["simulations"]["time_bins"].split()))),
                                       seed=config.getint("simulations", "training_seed"),
                                       keys=[],
                                       include_present_diversity=True)
    # save simulations
    res = dd.run_sim_parallel(training_set, config.getint("simulations", "n_CPUS"))
    print(res['features'].shape, res['labels'].shape)
    dd.save_simulations(res, config["simulations"]["sims_folder"], config["simulations"]["sim_name"] + "_" + now + "_training")


# next steps
"""
1. run simulations in parallel and save them
if n_training_simulations:
    training_set = dd.sim_settings_obj(bd_sim,
                                       fossil_sim,
                                       n_simulations=n_training_simulations,
                                       min_age=np.min(time_bins),
                                       max_age=np.max(time_bins),
                                       seed=training_seed,
                                       keys=[],
                                       include_present_diversity=True)
    # save simulations
    res = dd.run_sim_parallel(training_set, n_CPUS)
    print(res['features'].shape, res['labels'].shape)
    dd.save_simulations(res, output_path, outname + now + "_training")


2. model training ...

"""

















for key in config["simulations"]:  # lists settings included in a block
    print(key)

    try:
        os.mkdir('simulations/')
    except:
        pass
    try:
        os.mkdir('simulations/sqs_' + now)
    except:
        pass

# from the start of the model training script
wd = "./simulations"    # THESE LINES NEED EDITING - WILL NOT FUNCTION AS IS. USE ABS_PATH IDEA? HOW DID IT WORK IN DEEPDIVE FOR REVIEW? OR READ PATHS DIERCTLY FROM THE CONFIG FILE.
model_wd = "./model_training"

try:
    os.mkdir('model_training/')
except:
    pass


if __name__ == "__main__":
    ### SIMULATE MULTIPLE DATASETS ###
    # parallel simulations

    training_seed = config.getint("simulations", "training_seed")
    test_seed = config.getint("simulations", "test_seed")
    if training_seed == test_seed:
        print("Training and test seed are the same, please adjust one of them")

    if config.getint("simulations", "n_training_simulations"):
        if config.getint("simulations", "n_CPUS") > 1:
            list_args = list(np.arange(config.getint("simulations", "n_CPUS")))
            print("\nSimulating training data...")
            pool = multiprocessing.Pool()
            res = pool.map(run_sim, list_args, config=config)
            pool.close()

            features = []
            labels = []

            for i in range(config.getint("simulations", "n_CPUS")):
                features = features + res[i]['features']
                labels = labels + res[i]['labels']

        else:
            res = run_sim(0, config=config)
            features = res["features"]
            labels = res["labels"]

        Xt = np.array(features)
        Yt = np.array(labels)
        print(Xt.shape, Yt.shape)
        # save simulations
        out_path = config["simulations"]["out_path"]
        out_name = config["simulations"]["out_name"]
        np.save(os.path.join(out_path, out_name + "_features" + now + ".npy"), Xt)
        np.save(os.path.join(out_path, out_name + "_labels" + now + ".npy"), Yt)

        print("Training features saved as: \n", os.path.join(out_path, out_name + "_features" + now + ".npy"))
        print("Training labels saved as: \n", os.path.join(out_path, out_name + "_labels" + now + ".npy"))

        ### TEST DATASETS ###
    if config.getint("simulations", "n_test_simulations"):
        res = run_test_sim(0, config=config)

        features = res['features']
        labels = res['labels']

        Xt = np.array(features)
        Yt = np.array(labels)
        print(Xt.shape, Yt.shape)
        # save simulations
        np.save(os.path.join(out_path, out_name + "test_features" + now + ".npy"), Xt)
        np.save(os.path.join(out_path, out_name + "test_labels" + now + ".npy"), Yt)

        print("Test features saved as: \n", os.path.join(out_path, out_name + "test_features" + now + ".npy"))
        print("Test labels saved as: \n", os.path.join(out_path, out_name + "test_labels" + now + ".npy"))
        sim_settings = res['settings']
        dd.save_pkl(sim_settings, os.path.join(out_path, "test_sim_settings" + now + ".pkl"))

    print("\ndone.\n")


if __name__ == "__main__":
    nametag = 'base file name'

    feat_files = [
        '%s_features.npy' % (nametag)
    ]
    lab_file = '%s_labels.npy' % (nametag)

    output_names = ["rnn%s"]     # ['rnn%s' % date_tag]
    list_settings = get_model_settings()

    # import training data
    for i in range(len(feat_files)):
        feat_file = feat_files[i]

        for j in list_settings:
            j['feature_file'] = feat_file
            j['label_file'] = lab_file

        # run all jobs in parallel
        pool = multiprocessing.Pool(len(list_settings))
        pool.map(run_model_training, list_settings)
        pool.close()
