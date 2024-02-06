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
    feature_file, label_file = dd.run_sim_from_config(config)
else:
    feature_file, label_file = None
# Train a model
if config.getint("model_training", ""):
    dd.run_model_training_from_config(config, feature_file=feature_file, label_file=label_file)
# Simulate test set - just simulating, second test module that runs with models. Or maybe setting up training and test set
# don't need to be seperate? Add a default of automatically the same settings at the training set?? It would just change
# seed and number of simulations.
if config.getint("", ):  #### IF x2, sim training and if test under same conditions as simulate then read settings from simulate.
    dd.run_test_sim_from_config()




# Make diversity predictions




# next steps
"""
2. model training ...
options with multiprocessing?

3. testing?
similar to simulations process, saving outputs on model performance 

4. predictions
reading the new input file format, saving outputs and any graphs 

5. complete pipeline 
running the full thing and checking each block still functions independently, neaten up scripts. 
How will it be run in practice? 

6. empirical example

7. documentation and writing the application note

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
wd = "./simulations"
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
