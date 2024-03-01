import os
import deepdive as dd
from datetime import datetime
import configparser  # read the config file (".ini") created by the r function 'create_config()'

now = datetime.now().strftime('%Y%m%d')

config = configparser.ConfigParser()

# wd = "/Users/dsilvestro/Software/DeepDive-project/deepdive/test_deepdiveR/"
wd = "/Users/CooperR/Documents/GitHub/deep_dive/"
data_wd = "/Users/CooperR/Documents/GitHub/DeepDiveR/R/test_empirical_data/carnivora_analysis"
config_f = "carnivora.ini"
config.read(os.path.join(data_wd, config_f))
config.sections()  # see which blocks are listed in the config
# "simulations" in config  # to see if a block is present in the config, returns True/False
# config["simulations"]["eta"]  # to call settings


bd_sim, fossil_sim = dd.create_sim_obj_from_config(config)


# Test the simulation settings
sp_x = bd_sim.run_simulation(print_res=True)
sim = fossil_sim.run_simulation(sp_x)

# edit a setting in python after the fact
config["simulations"]["n_training_simulations"] = '10'
config["simulations"]["n_test_simulations"] = '10'

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
# Simulate test set - just sim, second test module that runs with models. Or maybe setting up training and test set
# don't need to be separate? Add a default of automatically the same settings at the training set?? It would just change
# seed and number of simulations.
# Train a model
if "model_training" in config.sections():
    dd.run_model_training_from_config(config, feature_file=feature_file, label_file=label_file)
# Predict diversity curves
if "empirical_predictions" in config.sections():
    results = dd.predict_from_config(config)
    predictions = results[0]
    res = results[1]


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
carnivores in the Cenozoic, global records from Faurby et al. Dispersal ability predicts evolutionary success among
mammalian carnivores.

7. documentation and writing the application note

"""
