import os
import pandas as pd
import numpy as np
import scipy.stats
import deepdive as dd
from deepdive import config_runner
import configparser 
import matplotlib
import matplotlib.pyplot as plt
import argparse
np.set_printoptions(suppress=True, precision=3)

p = argparse.ArgumentParser()
p.add_argument('config_file', metavar='<config file>', type=str, 
        help='Input config file')
p.add_argument('-wd', type=str, help='working directory', default=None, metavar="None")
p.add_argument('-cpu', type=int, help='number of CPUs', default=None, metavar="None")
p.add_argument('-lstm', type=int, help='nodes in LSTM layers', default=None, metavar="None", nargs='+')
p.add_argument('-dense', type=int, help='nodes in Dense layers', default=None, metavar="None", nargs='+')
p.add_argument('-train_set', type=str, help='training set file names', default=None, metavar="None")
p.add_argument('-test_set', type=str, help='test set file', default=None, metavar="None")
p.add_argument("-trained_model", default=None, type=str)
p.add_argument("-out_tag", default="", type=str)
p.add_argument("-calibrated", default=False, action='store_true')
p.add_argument("-plot_features", default=False, action='store_true')
p.add_argument('-n_sims', type=int, help='n simulations for plotting', default=100)

args = p.parse_args()


if args.plot_features:
    config_runner.sim_and_plot_features(args.config_file, wd=args.wd, CPU=args.cpu, n_sims=args.n_sims)
else:
    config_init = configparser.ConfigParser()
    config_runner.run_config(args.config_file, wd=args.wd, CPU=args.cpu,
                             train_set=args.train_set, test_set=args.test_set,
                             lstm=args.lstm, dense=args.dense, trained_model=args.trained_model,
                             out_tag=args.out_tag, calibrated=args.calibrated)

