# DeepDive 

You can install DeepDive either in your default Python environment, which will make the library available in any new Python instance, or within a virtual environment. The latter is particularly useful if you do not want DeepDive's dependencies to interfere with the libraries install in your default environment.

Before you start, make sure Python (v.3.10 - 3.12) is installed in your computer (you can find installers [here](https://www.python.org/downloads/)).
Note that the required [Tensorflow library](https://www.tensorflow.org) does not yet support Python v.3.13.  


## Installing DeepDive in your default Python environment 
You can then install DeepDive and all dependencies within your default Python environment running this command in a Terminal or Command prompt window:

```
python -m pip install git+https://github.com/DeepDive-project/deepdive@v.1.0
```

On Windows use `py` instead of `python`. Note that, if you have multiple versions of Python installed you might have to use e.g. `python3` or `python3.10` to select the desired version for the installation. 


## Installing DeepDive in a virtual environment
You can install DeepDive in a virtual environment to make sure all the compatible dependencies are included without affecting your system Python installation following the steps below.

1) Create a virtual environment typing in a terminal console (on Windows open a *command prompt* window and use `py` instead of `python`): 

```
python -m venv desired_path_to_env/dd_env
```  

2) Activate the virtual environment using on MacOS/UNIX: 

```
source desired_path_to_env/dd_env/bin/activate
```
or on Windows:

```
.\desired_path_to_env\dd_env\Scripts\activate
```  

3) Install DeepDive in the virtual env (on Windows use `py` instead of `python`) 

```
python -m pip install git+https://github.com/DeepDive-project/deepdive@v.1.0
```

The DeepDive library can now be imported after starting a Python console using:

```
>>> import deepdive as dd
>>> dd.__version__ # check version number
```


## Running a DeepDiveR configuration file

Configuration files (.ini) generated in [DeepDiveR](https://github.com/DeepDive-project/DeepDiveR) can be executed in the terminal window (MacOS and Linux) or command prompt (Windows) using the [`run_dd_config.py`](https://github.com/DeepDive-project/deepdive/blob/main/run_dd_config.py) script:

```
python run_dd_config.py your_path/config_file.ini
```

On Windows use `py` instead of `python`. Note that, if you have multiple versions of Python installed you might have to use e.g. `python3` or `python3.10` to select the desired version for the installation. 

The `run_dd_config.py` file referenced here reads variables from the config file and passes settings to wrapper functions that carry out the workflow in an automated pipeline. 
The program will automatically run an **autotuning** function that adjusts the parameters of the simulations to reflect the nature of the empirical data. A new config file will be saved in the same directory with the name tag `_autotuned.ini`. 
Details about the autotuning functions are provided in this [table](https://github.com/DeepDive-project/deepdive/blob/application_note/deepdive/deepdive_autotuning.md).
The output files of the DeepDive analysis (simulations, feature plots, and predictions) will be saved to the directory designated in the configuration file unless specified otherwise using the `-wd` command.

The working directory and number of CPUs can be directly adjusted in the command prompt. The number of simulations can be directly adjusted, for example to produce small batches of simulations and check the features are as expected: 

```
python run_dd_config.py your_path/config_file.ini -wd your_path -cpu 10 plot_features -n_sims 100
```



