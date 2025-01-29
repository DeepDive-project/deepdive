import sys
import numpy as np
import pandas as pd


class FeatureRescaler(object):
    def __init__(self, Xt, log_last=False):
        den2d = np.mean(Xt, axis=1)
        den1d = np.mean(den2d, axis=0)
        if len(np.where(den1d == 0)[0]):
            print("Warning - features %s is 0!" % np.where(den1d == 0))
            den1d[den1d == 0] = 1  # prevent divide-by-0 in case a features is always zero
        if log_last:
            # present diversity feature
            den1d[-1] = 1
        self.log_last = log_last
        self.den1d = den1d

    def feature_rescale(self, features):

        if len(features.shape) == 2:
            features = features.reshape((1, features.shape[0], features.shape[1]))

        x_r = features / self.den1d

        if self.log_last:
            x_r[:, 0, -1] = np.log(x_r[:, 0, -1] + 1)

        return x_r


def calculate_global_trajectory(species_space_time):
    # a 3D array with species, areas and time bins.
    temp = np.einsum('sat -> st', species_space_time)  # sum over areas
    x = temp + 0   # temp > 0
    x[temp > 1] = 1

    return np.einsum('st -> t', x)  # set to float and not integer for low res dating features


def calculate_sampled_turnover(species_space_time=None):
    species_space_time = np.random.binomial(1, 0.5, size=(5, 10))


def calculate_local_trajectory(species_space_time):
    # return a 2D matrix, areas by time bins
    x = species_space_time + 0  # turn n. of occs per species per area > 1 to 1
    x[x > 1] = 1
    return np.einsum('sat -> at', x.astype(float))


###---- start 1D FEATURES ----###
def time_bins_duration(sim):
    return sim['time_bins_duration']


# Count sampled species
def count_species(sim):  # same as global_fossil_trajectory
    global_fossil_trajectory = calculate_global_trajectory(sim["fossil_data"])
    return global_fossil_trajectory


# Count sampled localities, i.e. localities with fossils!
def count_sampled_localities(sim):
    sampled_localities = np.sum(sim["n_localities_w_fossils"], axis=0)
    return sampled_localities


# Count sampled occurrences
def count_occurrences(sim):
    # sum over localities? check what fossils record looks like.
    return np.einsum("sat -> t", sim["fossil_data"])  # make sure numpy is imported as np or will not run


# Total singletons
def count_singletons(sim):
    total_occurrences_per_sp = np.einsum("sat -> s", sim["fossil_data"])
    i_singleton_sp = np.where((total_occurrences_per_sp > 0) & (total_occurrences_per_sp <= 1))[0]
    access_singletons = sim["fossil_data"][i_singleton_sp, :, :]
    singletons = np.einsum("sat -> t", access_singletons)
    return singletons


# Count endemic species
def count_endemics(sim):
    areas_per_sp = np.einsum("sat -> sa", sim["fossil_data"])
    areas_per_sp_pres_abs = areas_per_sp + 0
    areas_per_sp_pres_abs[areas_per_sp_pres_abs > 0] = 1
    n_areas_per_species = np.sum(areas_per_sp_pres_abs, 1)
    i_endemic_sp = np.where((n_areas_per_species > 0) & (n_areas_per_species <= 1))[0]
    access_endemics = sim["fossil_data"][i_endemic_sp, :, :]
    z = np.zeros(access_endemics.shape)  # count per area, rather than by occurrences
    z[access_endemics > 0] = 1
    endemics = np.einsum("sat -> t", z)
    return endemics  # run a small simulation and check numbers are correct


def get_range_through_lineages(sim):
    global_records = np.einsum('sat -> st', sim['fossil_data'])
    # rm unsampled species
    record_sampled = global_records[np.where(np.sum(global_records, 1) > 0)[0], :]
    d_traj = np.zeros(record_sampled.shape)
    for sp in range(record_sampled.shape[0]):
        occ = record_sampled[sp, :].nonzero()[0]
        fa = np.max(occ)
        la = np.min(occ)
        d_traj[sp, np.arange(la, fa+1)] = 1
    return d_traj

def get_orig_ext_events(d_traj):
    # print("d_traj", d_traj)
    d_traj_rev = np.flip(d_traj, axis=1) # temporarily reverse time axis from old to young
    # print("d_traj_rev", d_traj_rev)
    d = np.diff(d_traj_rev, axis=1)
    # append first column
    events = np.hstack((np.expand_dims(d_traj_rev[:, 0], axis=1), d))
    # +1 : origination, -1: extinction |-> then reverse time axis again
    originations_per_bin = np.sum(events > 0, axis=0)[::-1]  # shape: (t)
    extinctions_per_bin = np.sum(events < 0, axis=0)[::-1]  # shape: (t)
    return originations_per_bin, extinctions_per_bin


# test get_orig_ext_events()
# rs = np.random.default_rng(123)
# x = rs.binomial(1, p=0.3, size= (10,1,15))
# x[8,0,0] = 0 # make one speices appear in the last bin
# np.squeeze(x)
# sim = {'fossil_data': x}
# z = get_range_through_lineages(sim).astype(int)
# get_orig_ext_events(z)

def get_range_through_diversity(sim, return_d_traj=False):
    d_traj = get_range_through_lineages(sim)
    ltt = np.sum(d_traj, 0)
    if return_d_traj:
        return ltt, d_traj
    else:
        return ltt
###---- end 1D FEATURES ----###


###---- start 2D FEATURES ----###
def sampled_localities_per_area_time(sim):
    msg = """N. of sampled localities cannot be established because the random binomial 
            draw used to sample species is not locality-specific - draw_fossils()"""
    sys.exit(msg)


def sampled_species_per_area_time(sim):
    return calculate_local_trajectory(sim['fossil_data'])
 

def n_occs_per_area_time(sim):
    return np.einsum('sat -> at', sim['fossil_data'])


def count_localities(sim):
    localities = sim["n_localities_w_fossils"]
    return localities

###---- end 2D FEATURES ----###


# EXTRACT FEATURES
def extract_features_1D(sim):
    properties = np.zeros((sim["n_bins"], 9))
    properties[:, 0] = count_species(sim)
    properties[:, 1] = count_occurrences(sim)
    properties[:, 2] = count_singletons(sim)
    properties[:, 3] = count_endemics(sim)
    properties[:, 4] = time_bins_duration(sim)
    properties[:, 5], d_traj = get_range_through_diversity(sim, return_d_traj=True)
    properties[:, 6] = count_sampled_localities(sim)
    a, b = get_orig_ext_events(d_traj)
    properties[:, 7] = a
    properties[:, 8] = b
    return properties


def extract_features_2D(sim):
    p1 = sampled_species_per_area_time(sim)
    p2 = n_occs_per_area_time(sim)
    p3 = count_localities(sim)
    properties = np.vstack((p1, p2, p3)).T
    # properties.shape = (n_time_bins, n_areas x 2)
    return properties


def extract_sim_features(sim, include_present_div=False):
    f1d = extract_features_1D(sim)
    f2d = extract_features_2D(sim)
    if include_present_div:
        present_div = np.zeros((f1d.shape[0], 1))
        present_div[0] += sim['global_true_trajectory'][0]
        feat = np.hstack((f1d, f2d, present_div))
    else:
        feat = np.hstack((f1d, f2d))
    return feat



def normalize_features(Xt, log_last=False):
    den2d = np.mean(Xt, axis=1)
    den1d = np.mean(den2d, axis=0)
    if len(np.where(den1d == 0)[0]):
        print("Warning - features %s is 0!" % np.where(den1d == 0))
        den1d[den1d == 0] = 1 # prevent divide-by-0 in case a features is always zero

    if log_last:
        # present diversity feature
        den1d[-1] = 1
        def feature_rescaler(x):
            x_r = x / den1d
            x_r[-1] = np.log(x_r[-1] + 1)
            return x_r
    else:
        def feature_rescaler(x):
            x_r = x / den1d
            return x_r

    Xt_r = feature_rescaler(Xt)

    return Xt_r, feature_rescaler


def normalize_labels(Yt, rescaler=0, log=False):
    Yt_r = Yt + 0
    if rescaler == 0:
        Yt_r = Yt_r / np.mean(Yt_r)
    else:
        Yt_r = Yt_r / rescaler
    if log:
        Yt_r = np.log(Yt_r + 1)
    return Yt_r


def get_features_names(n_areas, include_present_div=False, as_dataframe=False, area_names=None):
    global_features = ["n_species", "n_occs", "n_singletons", "n_endemics",
                       "time_bin_duration", "range_through_div", "n_localities",
                       "origination_events", "extinction_events"]

    if area_names is None:
        sp_area = ["n_species_area_%s" % i for i in range(1, 1 + n_areas)]
        occs_area = ["n_occs_area_%s" % i for i in range(1, 1 + n_areas)]
        loc_area = ["n_locs_area_%s" % i for i in range(1, 1 + n_areas)]
    else:
        sp_area = ["n_species_%s" % i for i in area_names]
        occs_area = ["n_occs_%s" % i for i in area_names]
        loc_area = ["n_locs_%s" % i for i in area_names]


    features_names = global_features + sp_area + occs_area + loc_area
    if include_present_div:
        features_names = features_names + ["present_diversity"]

    if as_dataframe:
        features_names = pd.DataFrame(zip(range(len(features_names)), features_names))
        features_names.columns = ["Index", "Feature"]

    return features_names
