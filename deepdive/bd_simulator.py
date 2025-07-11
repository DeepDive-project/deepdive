import sys
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from collections.abc import Iterable
from .feature_extraction import *
from .utilities import get_rnd_gen
SMALL_NUMBER = 1e-10

class custom_bd():
    def transform_rates(self, timesL, timesM, L, M):
        return timesL, timesM, L, M


class bd_simulator():
    def __init__(self,
                 s_species=1,  # number of starting species (can be a range)
                 total_species=[100, 1000],  # min/max size data set
                 min_extinct_species=0,  # minimum number of extinct lineages allowed
                 min_extant_sp=0,
                 max_extant_sp=np.inf,
                 pr_extant_clade=None,
                 root_r=[30., 100],  # range root ages
                 rangeL=[0.2, 0.5],
                 rangeM=[0.2, 0.5],
                 rangeI=None,
                 scale=100.,
                 p_mass_extinction=[0, 0.00924],
                 magnitude_mass_ext=[0.8, 0.95],
                 fixed_mass_extinction=None, # list of ME ages
                 p_mass_speciation=0,
                 magnitude_mass_sp=[0.5, 0.95],
                 poiL=3,
                 poiM=3,
                 p_constant_bd=0.05,
                 p_equilibrium=0.1,
                 p_dd_model=0,
                 dd_K=100,
                 dd_maxL=None, # max speciation rate
                 log_uniform_rates=False,
                 log_uniform_species=False,
                 survive_age_condition=None,
                 seed=None,
                 vectorize=False,
                 max_iter=10000,
                 bd_alter_obj = None):
        self.s_species = s_species
        self.total_species = total_species
        self.minSP = np.min(total_species)
        self.maxSP = np.max(total_species)
        self.min_extinct_species = min_extinct_species
        self.min_extant_sp = min_extant_sp
        self.max_extant_sp = max_extant_sp
        self.pr_extant_clade = pr_extant_clade
        self.root_r = root_r
        self.rangeL = rangeL
        self.rangeM = rangeM
        self.rangeI = rangeI
        self.scale = scale
        self.p_mass_extinction = p_mass_extinction
        self.magnitude_mass_ext = np.sort(magnitude_mass_ext)
        self.p_mass_speciation = p_mass_speciation
        self.fixed_mass_extinction = fixed_mass_extinction
        self.magnitude_mass_sp = np.sort(magnitude_mass_sp)
        self.poiL = poiL
        self.poiM = poiM
        self.p_constant_bd = p_constant_bd
        self.p_equilibrium = p_equilibrium
        self.p_dd_model = p_dd_model
        self.dd_K = dd_K
        self.dd_maxL = dd_maxL
        self.log_uniform_rates = log_uniform_rates
        self.log_uniform_species = log_uniform_species
        self.vectorize = vectorize
        self.max_iter = max_iter
        self.survive_age_condition = survive_age_condition
        self._rs = get_rnd_gen(seed)
        self._bd_alter_obj = bd_alter_obj
        if log_uniform_species:
            self.dd_K = np.log(self.dd_K)
            self.s_species = np.log(self.s_species)

    def reset_seed(self, seed):
        self._rs = get_rnd_gen(seed)

    def simulate(self, L, M, timesL, timesM, root, dd_model=False, verbose=False):
        ts = list()
        te = list()
        L, M, root = L / self.scale, M / self.scale, int(root * self.scale)
        I_events = 0

        # print(L, M, root)

        if dd_model:
            M = self._rs.uniform(np.min(self.rangeM), np.max(self.rangeM), 1)  / self.scale
            if isinstance(self.dd_K, Iterable):
                if self.log_uniform_species:
                    k_cap_vec = np.exp(self._rs.uniform(np.min(self.dd_K),
                                                        np.max(self.dd_K),
                                                        len(timesL) - 1)).astype(int)
                else:
                    k_cap_vec = self._rs.integers(np.min(self.dd_K), np.max(self.dd_K), len(timesL) - 1)
            else:
                k_cap_vec = np.ones(len(timesL) - 1) * (self.dd_K)
            # print("k_cap_vec", k_cap_vec)

        if isinstance(self.p_mass_extinction, Iterable):
            mass_extinction_prob = self._rs.choice(self.p_mass_extinction) / self.scale
            mass_speciation_prob = self._rs.choice(self.p_mass_speciation) / self.scale
        else:
            mass_extinction_prob = self.p_mass_extinction / self.scale
            mass_speciation_prob = self.p_mass_speciation / self.scale

        if self.rangeI is not None:
            if isinstance(self.rangeI, Iterable):
                immigration_rate = self._rs.uniform(np.min(self.rangeI), np.max(self.rangeI), 1) / self.scale
            else:
                immigration_rate = self.rangeI / self.scale


        if isinstance(self.s_species, Iterable):
            if self.log_uniform_species:
                s_species = int(np.exp(
                    self._rs.uniform(self.s_species[0],
                                     self.s_species[1])))
            else:
                if self.s_species[0] == self.s_species[1]:
                    self.s_species[1] = self.s_species[0] + 1
                s_species = self._rs.integers(self.s_species[0], self.s_species[1])
            ts = list(np.zeros(s_species) + root)
            te = list(np.zeros(s_species))
        else:
            for i in range(self.s_species):
                ts.append(root)
                te.append(0)

        done = True
        if self.fixed_mass_extinction is not None:
            done = False

        if self.survive_age_condition is not None:
            done = False

        for t in range(root, 0):  # time
            if self.rangeI is not None:
                if self._rs.random() < immigration_rate:
                    te.append(0)  # add species
                    ts.append(t)  # sp time
                    I_events += 1

            if not dd_model:
                for j in range(len(timesL) - 1):
                    if -t / self.scale <= timesL[j] and -t / self.scale > timesL[j + 1]:
                        l = L[j]
                for j in range(len(timesM) - 1):
                    if -t / self.scale <= timesM[j] and -t / self.scale > timesM[j + 1]:
                        m = M[j]

            # if t % 100 ==0: print t/scale, -times[j], -times[j+1], l, m
            TE = len(te)
            if TE > self.maxSP:
                break
            ran_vec = self._rs.random(TE)
            te_extant = np.where(np.array(te) == 0)[0]

            no = self._rs.random(2)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species

            if no_extant_lineages == 0 and self.rangeI is None:
                # stop loop over time when clade is extinct
                return -np.array(ts) / self.scale, -np.array(te) / self.scale, done

            if dd_model:
                for j in range(len(timesL) - 1):
                    if -t / self.scale <= timesL[j] and -t / self.scale > timesL[j + 1]:
                        k_cap = k_cap_vec[j]


                m = M[0]
                l = m * k_cap / np.max([1, no_extant_lineages])
                if self.dd_maxL is not None:
                    l = np.min([l, self.dd_maxL / self.scale])
                # print("DD", l, m, no_extant_lineages)

            if self.survive_age_condition is not None:
                delta_from_condition = np.min(
                    np.abs(np.abs(t) - np.array(self.survive_age_condition) * self.scale))
                if delta_from_condition < 1:
                    done = True

            if self.fixed_mass_extinction is not None:
                delta_from_condition = np.min(
                    np.abs(np.abs(t) - np.array(self.fixed_mass_extinction) * self.scale))

                if delta_from_condition < 1:
                    no[0] = 0
                    # print(np.abs(t), np.array(self.fixed_mass_extinction) * self.scale,
                    #       self.scale , no[0])
                    if verbose:
                        print(np.abs(t), np.array(self.fixed_mass_extinction) * self.scale)
                        print("Mass extinction", t / self.scale, mass_extinction_prob, no[0])
                    # increased loss of species: increased ext probability for this time bin
                    m = self._rs.uniform(self.magnitude_mass_ext[0], self.magnitude_mass_ext[1])
                    # if the clade reaches the fixed mass extinction then the simulation is successful
                    done = True

            if no[0] < mass_extinction_prob and no_extant_lineages > 10 and t > root:  # mass extinction condition
                if verbose:
                    print("Mass extinction", t / self.scale, mass_extinction_prob, no[0])
                # increased loss of species: increased ext probability for this time bin
                m = self._rs.uniform(self.magnitude_mass_ext[0], self.magnitude_mass_ext[1])
            if no[1] < mass_speciation_prob and t > root:
                l = self._rs.uniform(self.magnitude_mass_sp[0], self.magnitude_mass_sp[1])

            if self.vectorize:
                if m == 1: # all go extinct, no speciation events possible
                    l = 0
                te = np.array(te)
                ext_species = np.where((ran_vec > l) & (ran_vec < (l + m)) & (te == 0))[0]
                te[ext_species] = t

                rr = ran_vec[te_extant]
                n_new_species = len(rr[rr < l])
                te = list(te) + list(np.zeros(n_new_species))
                ts = ts + list(np.zeros(n_new_species) + t)

            else:
                for j in te_extant:  # extant lineages
                    # if te[j] == 0:
                    ran = ran_vec[j]
                    if ran < l:
                        te.append(0)  # add species
                        ts.append(t)  # sp time
                    elif ran > l and ran < (l + m):  # extinction
                        te[j] = t


        if verbose:
            print("Immigration events:", I_events)

        return -np.array(ts) / self.scale, -np.array(te) / self.scale, done

    def get_random_settings(self, root):
        root = np.abs(root)
        timesL_temp = [root, 0.]
        timesM_temp = [root, 0.]

        rr = self._rs.random(2)

        if rr[0] < self.p_constant_bd:
            nL = 0
            nM = 0
            timesL = np.array(timesL_temp)
            timesM = np.array(timesM_temp)
        elif rr[1] < self.p_equilibrium:
            nL = self._rs.poisson(self.poiL)
            nM = nL
            shift_time_L = self._rs.uniform(0, root, nL)
            shift_time_M = shift_time_L
            timesL = np.sort(np.concatenate((timesL_temp, shift_time_L), axis=0))[::-1]
            timesM = np.sort(np.concatenate((timesM_temp, shift_time_M), axis=0))[::-1]
        else:
            nL = self._rs.poisson(self.poiL)
            nM = self._rs.poisson(self.poiM)
            shift_time_L = self._rs.uniform(0, root, nL)
            shift_time_M = self._rs.uniform(0, root, nM)
            timesL = np.sort(np.concatenate((timesL_temp, shift_time_L), axis=0))[::-1]
            timesM = np.sort(np.concatenate((timesM_temp, shift_time_M), axis=0))[::-1]

        if self.log_uniform_rates:
            L = np.exp(self._rs.uniform(np.log(np.min(self.rangeL)),
                                         np.log(np.max(self.rangeL)),
                                         nL + 1))
            M = np.exp(self._rs.uniform(np.log(np.min(self.rangeM)),
                                         np.log(np.max(self.rangeM)),
                                         nM + 1))
        else:
            L = self._rs.uniform(np.min(self.rangeL), np.max(self.rangeL), nL + 1)
            M = self._rs.uniform(np.min(self.rangeM), np.max(self.rangeM), nM + 1)

        if rr[1] < self.p_equilibrium:
            indx_equilibrium = self._rs.choice(range(len(L)), size=self._rs.integers(1, len(L)+1))
            M[indx_equilibrium] = L[indx_equilibrium]

        # M[0] = self._rs.uniform(0,.1*L[0])

        return timesL, timesM, L, M

    def run_simulation(self, print_res=False, return_bd_settings=False, fixed_bd_settings=None):
        LOtrue = [0]
        n_extinct = -0
        n_extant = -0
        done = False
        if self.p_dd_model > self._rs.random():
            dd_model = True
        else:
            dd_model = False

        if self.pr_extant_clade is not None:
            if self._rs.random() < self.pr_extant_clade:
                min_extant = np.maximum(1, self.min_extant_sp) + 0 # clade is extant
                max_extant = self.max_extant_sp + 0
            else:
                min_extant = 0
                max_extant = 0
        else:
            min_extant = self.min_extant_sp + 0
            max_extant = self.max_extant_sp + 0

        counter = 0
        while (len(LOtrue) < self.minSP or
               len(LOtrue) > self.maxSP or
               n_extinct < self.min_extinct_species or
               n_extant < min_extant or
               n_extant > max_extant or
               done is False):

            if counter > 100:
                if self._rs.random() < 0.5:
                    min_extant = self.min_extant_sp
                    max_extant = self.max_extant_sp
                else:
                    dd_model = False
            if counter == self.max_iter:
                print(f"""
                Warning: DeepDive failed to run a birth-death simulations after {counter} iterations. 
                Consider changing the simulation settings.              
                """)

            if isinstance(self.root_r, Iterable):
                root = -self._rs.uniform(np.min(self.root_r), np.max(self.root_r))  # ROOT AGES
            else:
                root = -self.root_r + 0

            if fixed_bd_settings is not None:
                [timesL, timesM, L, M] = fixed_bd_settings
            else:
                timesL, timesM, L, M = self.get_random_settings(root)
            if self._bd_alter_obj is not None:
                timesL, timesM, L, M = self._bd_alter_obj.transform_rates(timesL, timesM, L, M)
                dd_model = False
                # print("timesL, timesM, L, M ", timesL, timesM, L, M )
            FAtrue, LOtrue, done = self.simulate(L, M, timesL, timesM, root, dd_model=dd_model, verbose=print_res)
            n_extinct = len(LOtrue[LOtrue > 0]) + 0
            n_extant = len(LOtrue[LOtrue == 0]) + 0
            # print('prm', n_extinct, len(LOtrue), n_extant, min_extant, max_extant, self.min_extant_sp, self.max_extant_sp)
            counter += 1

        # print("timesL, timesM, L, M ", timesL, timesM, L, M)
        ts_te = np.array([FAtrue, LOtrue])
        # print_res = True
        if print_res:
            print("L", L, "M", M, "tL", timesL, "tM", timesM)
            print("N. species", len(LOtrue))
            max_standin_div = np.max([len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i]) for i in range(int(max(FAtrue)))]) / 80

            ltt = ""
            for i in range(int(max(FAtrue))):
                n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                ltt += "\n%s\t%s\t%s" % (i, n, "*" * int(n / max_standin_div))
            print(ltt)
        if return_bd_settings:
            res_dict = {"L": L, "M": M,
                        "tL": timesL, "tM": timesM,
                        "n_extinct": n_extinct,
                        "n_extant": n_extant
                        }
            return ts_te.T, res_dict
        return ts_te.T

    def reset_s_species(self, s):
        self.s_species = s

    def set_bd_alter_obj(self, f: custom_bd):
        self._bd_alter_obj = f

