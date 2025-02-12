# DeepDive 

You can install DeepDive either in your default Python environment, which will make the library available in any new Python instance, or within a visrtual environemnt. The latter is partiucularly useful if you do not want DeepDive's dependencies to interfere with the libraries installl in your default environment.

## Installing DeepDive in your default Python environment
First, make sure Python (v.3.10 or above) is installed in your computer (you can find installers [here](https://www.python.org/downloads/)).    
You can then install DeepDive and all dependencies within your deafult Python environment running this command in a Terminal or Command prompt window:

```
python -m pip install git+https://github.com/DeepDive-project/deepdive
```

On Windows use `py` instead of `python`. Note that, if you have multiple versions of Python installed you might have to use e.g. `python3` or `python3.10` to select the desired version for the installation. 


## Installing DeepDive in a virtual environment
You can install DeepDive in a virtual environment to make sure all the compatible dependencies are included without affecting your system Python installation following the steps below:  
Before you start, make sure Python (v.3.10 or above) is installed in your computer (you can find installers [here](https://www.python.org/downloads/)).    

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

3) Install DeepDive in the virtual env (on Windows use `py` instead of `python`) after replacing `your_path` with the correct path to the `deepdive` directory:

```
python -m pip install --upgrade pip # check that pip is up to date
python -m pip install your_path_to_deepdive/.
```

The DeepDive library can now be imported after starting a Python console using:

```
>>> import deepdive as dd
>>> dd.__version__ # check version number
```


## Running a DeepDiveR configuration file

Configuration files (.ini) generated in DeepDiveR can be executed in the terminal window (MacOS and Linux) or command prompt (Windows):

```
python run_dd_config.py your_path/config_file.ini
```

On Windows use `py` instead of `python`. Note that, if you have multiple versions of Python installed you might have to use e.g. `python3` or `python3.10` to select the desired version for the installation. 

The `run_dd_config.py` file referenced here reads variables from the config file and passess settings to wrapper functions that carry out the workflow in an automated pipeline. 
The program will automatically run an **autotuning** function that adjusts the parameters of the simulations to reflect the nature of the empirical data. A new config file will be saved in the same directory with the name tag `_autotuned.ini`. 
Details about the autotuning functions are provided in this [table](https://github.com/DeepDive-project/deepdive/blob/application_note/deepdive/deepdive_autotuning.md).
The output files of the DeepDive analysis (simulations, feature plots, and predictions) will be saved to the directory designated in the configuration file unless specified otherwise using the `-wd` command.

The working directory and numbr of CPUs can be direclty adjusted in the command prompt. The number of simulations can be directly adjusted, for example to produce small batches of simulations and check the features are as expected: 

```
python run_dd_config.py your_path/config_file.ini -wd your_directory -cpu 10 plot_features -n_sims 100
```



