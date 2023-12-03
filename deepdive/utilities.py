import copy
import glob
import os
import pickle
import sys

import numpy as np
import pandas
import pandas as pd
import scipy.stats

from .feature_extraction import extract_sim_features

def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()


def unique_unsorted(a_tmp):
    a = copy.deepcopy(a_tmp)
    indx = np.sort(np.unique(a, return_index=True)[1])
    u = a_tmp[indx]
    return u


def load_pkl(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        import pickle5
        with open(file_name, 'rb') as f:
            return pickle5.load(f)


def save_pkl(obj, out_file):
    with open(out_file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def preservation_rate_mle(all_br, all_occs, sp_x, n):
    for i in range(len(all_br)):
        l_range = np.linspace(0, 2 * all_occs[i] / all_br[i], 10000)
        lik = []
        for l in l_range:
            lik.append(np.sum(scipy.stats.poisson.logpmf(n, l) - l * (sp_x[:, 0] - sp_x[:, 1])))
        lik = np.array(lik)
        mle = l_range[np.where(lik == np.max(lik))[0]][0]
        print("max likelihood estimated rate", np.round(mle, 3))


def get_confidence_intervals(prediction, confidence_interval):
    lower_lim = np.quantile(prediction, 0.5 - confidence_interval / 2, axis=0)
    upper_lim = np.quantile(prediction, 0.5 + confidence_interval / 2, axis=0)
    return np.array([lower_lim, upper_lim])


def set_area_constraints(sp_x,
                         n_time_bins,
                         area_tbl: pandas.DataFrame,
                         mid_time_bins
                         ):
    """
    :param sp_x: output of bd_sim.run_simulation()
    :param n_time_bins: number of time bins
    :param area_tbl: data.frame (rows: n_areas, cols: [area_id, start, end]) if -1 start at the beginning or end at time 0
    :param mid_time_bins: fossil_sim.mid_time_bins

    :return: input for fossil_sim.set_carrying_capacity_multiplier()
    """
    n_species = sp_x.shape[0]
    area_array = area_tbl.to_numpy()
    n_areas = area_array.shape[0]
    carrying_capacity_mul_species_origin = np.ones((n_species, n_areas))
    carrying_capacity_mul_sp_area_time = np.ones((n_species, n_areas, n_time_bins))

    for area_i in range(n_areas):
        start = area_array[area_i,1]
        end = area_array[area_i, 2]
        if start > -1:
            carrying_capacity_mul_species_origin[np.where(sp_x[:, 0] > start)[0], area_i] *= 0
            carrying_capacity_mul_sp_area_time[:, area_i, np.where(mid_time_bins > start)[0]] *= 0
        if end > -1:
            carrying_capacity_mul_species_origin[np.where(sp_x[:, 0] < end)[0], area_i] *= 0
            carrying_capacity_mul_sp_area_time[:, area_i, np.where(mid_time_bins < end)[0]] *= 0


    return carrying_capacity_mul_species_origin, carrying_capacity_mul_sp_area_time



def prep_dd_input(wd,
                  bin_duration_file='hr_t_bins.csv',  # from old to recent, array of shape (t)
                  locality_file='hr_localities.csv',  # array of shape (a, t)
                  locality_dir='Locality',
                  taxon_dir='Genus_occurrences',
                  hr_time_bins=None,  # array of shape (t)
                  lr_locality_file=None,  # array of shape (a, low_res_t)
                  lr_time_bins=None,
                  no_age_u=False,
                  replicate=None,
                  rescale_by_n_bins=True,
                  debug=False
                  ):
    # time bin file
    bin_durations = pd.read_csv(os.path.join(wd, bin_duration_file))  # Time bins duration
    bin_durations = np.array(bin_durations).flatten()  # FROM OLD TO RECENT
    bin_durations_rev = bin_durations[::-1]  # FROM RECENT TO OLD

    # occs files
    if no_age_u is False:
        occs_files = np.sort(glob.glob(os.path.join(wd, taxon_dir, "hr_*.csv")))
    if no_age_u is True:
        occs_files = np.sort(glob.glob(os.path.join(wd, taxon_dir, str(replicate) + "_*.csv")))
    occurrences = np.array([pd.read_csv(f) for f in occs_files])
    occs = np.transpose(occurrences, axes=(1, 0, 2))
    hr_occs_rev = np.flip(occs, axis=2)  # FROM RECENT TO OLD
    # locality file
    localities = pd.read_csv(os.path.join(wd, locality_dir, locality_file)).to_numpy()  # localities
    localities_rev = np.flip(localities, axis=1)  # FROM RECENT TO OLD

    sim = {
        'n_bins': hr_occs_rev.shape[2],
        'fossil_data': hr_occs_rev,
        'n_localities_w_fossils': localities_rev,
        'time_bins_duration': bin_durations_rev,
    }


    # Low res occs files
    if lr_locality_file is not None:
        raise NotImplementedError("\n Function not available!")
    else:
        # High res features only
        # print(hr_occs_rev.shape, localities_rev.shape, bin_durations_rev.shape)
        features = extract_sim_features(sim)
        info = None

    # next: rescale features using rescaler and run predictions with Dropout

    if debug:
        # check that after diluting the counts the totals remain unchanged
        print(np.sum(info['lr_foss_data']) == np.sum(lr_occs_rev))  # total occurrences
        print(np.sum(info['lr_loc_data']) == np.sum(lr_localities_rev))  # total localities
        # total occurrences vs summed counts in the low res features:
        print(np.sum(info['lr_foss_data']) / np.sum(info['sim_features_lr'][:, 1]))
        print(np.sum(info['sim_features_hr'][:, 1]) / np.sum(hr_occs_rev))  # hr occs counts
        print(np.sum(localities), np.sum(info['sim_features_hr'][:, 6]))  # hr localities
        print(info['time_bins_lr_duration'])

    return features, info


def predict(features,
            model,
            feature_rescaler=None,
            drop_hr_features=False,
            n_predictions=1,
            dropout=True,
            ):
    # next: rescale features using rescaler and run predictions with Dropout
    if feature_rescaler is not None:
        features_rescaled = feature_rescaler(features)  # FROM RECENT TO OLD
    else:
        features_rescaled = features
    if len(features.shape) == 2:
        dd_input = features_rescaled.reshape((1, features.shape[0], features.shape[1]))
    elif len(features.shape) == 3:
        dd_input = features_rescaled
    else:
        print("Cannot reshape feature file")
        return "Error"
    if drop_hr_features:
        # leave only low res data!
        dd_input = dd_input[:, :, 22:]

    # predictions
    if len(features.shape) == 2:
        Ytest_pred = np.array([np.array(model(dd_input, training=dropout)).flatten() for _ in range(n_predictions)])
    else:
        Ytest_pred = np.array([np.array(model(dd_input, training=dropout)).reshape(dd_input.shape[:2]) for _ in range(n_predictions)])
        # shape = (n_predictions, n_instances, n_time_bins)
    return Ytest_pred


def calcCI(data, level=0.95):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])
