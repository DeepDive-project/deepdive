import multiprocessing
from collections.abc import Iterable
import numpy as np
import pandas as pd
import os
from .utilities import print_update, save_pkl, set_area_constraints
from .feature_extraction import extract_sim_features, get_features_names
from .plots import plot_feature_hists
from multiprocessing import get_context

class sim_settings_obj():
    def __init__(self,
                 bd_sim,
                 fossil_sim,
                 n_simulations,
                 min_age,
                 max_age,
                 seed=None,
                 keys=None,
                 verbose=False,
                 include_present_diversity=False,
                 area_constraint=None
                 ):
        self.bd_sim = bd_sim
        self.fossil_sim = fossil_sim
        self.n_simulations = n_simulations
        self.min_age = min_age
        self.max_age = max_age

        if seed is None:
            self.seed = np.random.randint(1000, 9999)
        else:
            self.seed = seed
        if keys is None:
            self.keys = []
        else:
            self.keys = keys
        self.verbose = verbose
        self.include_present_diversity = include_present_diversity
        self.area_constraint = area_constraint



def run_sim(args):
    if isinstance(args, Iterable):
        [rep, settings_obj] = args
    else:
        rep, settings_obj = 0, args
    batch_features = []
    batch_labels = []

    # simulate training data
    sim_settings = []
    bd_sim = settings_obj.bd_sim
    fossil_sim = settings_obj.fossil_sim
    # reset seed based on rep, if parallelized
    bd_sim.reset_seed(settings_obj.seed + rep)
    fossil_sim.reset_seed(settings_obj.seed + rep)
    rs = np.random.default_rng(settings_obj.seed + rep)

    for i in range(settings_obj.n_simulations):
        if i % 1 == 0 and rep == 0:
            if settings_obj.verbose is False:
                print_update("Running simulation %s of %s" % (i + 1, settings_obj.n_simulations))

        sp_x = bd_sim.run_simulation(print_res=settings_obj.verbose)

        ### area constraints
        area_tbl = None
        # print("settings_obj.area_constraint ", settings_obj.area_constraint )
        if settings_obj.area_constraint is not None:
            fossil_sim.reset()
            fossil_sim.bd_sim_setup(sp_x, min_age=settings_obj.min_age, max_age=settings_obj.max_age)

            area_tbl = settings_obj.area_constraint['min_age'] + 0
            rnd_d = rs.uniform(settings_obj.area_constraint['min_age'][:,1],
                                      settings_obj.area_constraint['max_age'][:,1])

            area_tbl[:,1] = rnd_d

            rnd_d = rs.uniform(settings_obj.area_constraint['min_age'][:,2],
                                      settings_obj.area_constraint['max_age'][:,2])

            area_tbl[:, 2] = rnd_d
        if area_tbl is not None:
            # print(sp_x)
            area_tbl = pd.DataFrame(area_tbl)
            # print("area_tbl:", area_tbl)
            # area constraints
            # data.frame (rows: n_areas, cols: [area_id, start, end]) if -1 start at the beginning or end at time 0
            # the dataframe can also be read from a file e.g. using pd.read_csv()

            c1, c2 = set_area_constraints(sp_x=sp_x,
                                          n_time_bins=fossil_sim.n_bins,
                                          area_tbl=area_tbl,
                                          mid_time_bins=fossil_sim.mid_time_bins)

            # print("c1:", c1.shape)
            # print("c2:", c2.shape)
            # quit()
            fossil_sim.set_carrying_capacity_multiplier(m_species_origin=c1, m_sp_area_time=c2)

        sim = fossil_sim.run_simulation(sp_x, min_age=settings_obj.min_age, max_age=settings_obj.max_age)
        sim_features = extract_sim_features(sim,
                                            include_present_div=settings_obj.include_present_diversity)
        sim_y = sim['global_true_trajectory']
        batch_features.append(sim_features)
        batch_labels.append(sim_y)

        s = {key: sim[key] for key in settings_obj.keys}
        sim_settings.append(s)
    if rep == 0:
        print("\ndone.\n")

    res = {'features': np.array(batch_features),
            'labels': np.array(batch_labels),
            'settings': sim_settings}
    return res

def run_sim_parallel(training_set: sim_settings_obj, n_CPUS):
    if n_CPUS == 1:
        print("\nSimulating data...")
        res = [run_sim([0, training_set])]
    else:
        training_args = [[i, training_set] for i in range(n_CPUS)]
        print("\nSimulating data (parallelized on %s CPUs)..." % n_CPUS)

        p = get_context("fork").Pool(n_CPUS)
        res = p.map(run_sim, training_args)
        p.close()
        # pool = multiprocessing.Pool()
        # res = pool.map(run_sim, training_args)
        # pool.close()

    features = []
    labels = []
    for i in range(n_CPUS):
        features = features + list(res[i]['features'])
        labels = labels + list(res[i]['labels'])

    Xt = np.array(features)
    Yt = np.array(labels)

    res = {'features': Xt, 'labels': Yt}
    return res

def save_simulations(res, output_path, outname, return_file_names=False):
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    np.save(os.path.join(output_path, outname + "_features" + ".npy"), res['features'])
    np.save(os.path.join(output_path, outname + "_labels" + ".npy"), res['labels'])
    print("Features saved as: \n", os.path.join(output_path, outname + "_features" + ".npy"))
    print("Labels saved as: \n", os.path.join(output_path, outname + "_labels" + ".npy"))

    if 'settings' in res.keys():
        save_pkl(res['settings'], os.path.join(output_path, outname + "_sim_settings.pkl"))
        print("Settings saved as: \n", os.path.join(output_path, outname + "_sim_settings.pkl"))
    if return_file_names:
        return  (os.path.join(output_path, outname + "_features" + ".npy"),
                 os.path.join(output_path, outname + "_labels" + ".npy"))



def compare_features(empirical_features,
                     test_set: sim_settings_obj = None,
                     test_res= None,
                     test_set_features=None,
                     return_simulations=False,
                     log_occurrences=False,
                     n_areas=None,
                     wd="",
                     output_name=None,
                     show=True):
    if test_set_features is None:
        if test_res is None:
            test_res = run_sim(test_set)
        test_set_features = test_res['features']

    if n_areas is None:
        n_areas = test_set.fossil_sim.n_areas

    features_names = get_features_names(n_areas)
    plot_feature_hists(test_features=test_set_features,
                       empirical_features=empirical_features,
                       show=show,
                       wd=wd, output_name=output_name,
                       features_names=features_names,
                       log_occurrences=log_occurrences)
    if return_simulations:
        return test_res
