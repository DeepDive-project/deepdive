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
args = p.parse_args()

config_init = configparser.ConfigParser()
config_runner.run_config(args.config_file, wd=args.wd, CPU=args.cpu)