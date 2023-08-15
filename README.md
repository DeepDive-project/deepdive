# DeepDive 

### Installing DeepDive in a virtual environment
You can install DeepDive in a virtual environment to make sure all the compatible dependencies are included without affecting your system Python installation following the steps below:  
Before you start, make sure Python (v.3.9 or above) is installed in your computer  

1. Create a virtual environment typing in a terminal console (on Windows use `py` instead of `python`): 
```
python -m venv desired_path_to_env/dd_env
```  
2. activate the virtual environment using on MacOS/UNIX: 
```
source desired_path_to_env/dd_env/bin/activate
```
or on Windows:
```
.\desired_path_to_env\dd_env\Scripts\activate
```  
3. install DeepDive in the virtual env (on Windows use `py` instead of `python`):
```
python -m pip install path_to_deepdive
```

The DeepDive library can now be imported after starting a Python console using:

```
>>> import deepdive as dd
>>> dd.__version__ # check version number
```

---
DeepDive is a Python library compatible with Python v. 3.8 or greater. Running the library requires the following dependencies:

```
numpy~=1.22.3
matplotlib~=3.5.2
pandas~=1.4.3
scipy~=1.8.1
tensorflow~=2.8.0
seaborn~=0.11.2

```

These dependencies can be installed using the `requirements.txt` file provided with DeepDive:

```
pip install -r requirements.txt
```

The DeepDive library can then be imported from a local Python console using:

```
>>> import deepdive as dd
>>> dd.__version__

```

We provide scripts showing how to use DeepDive to simulate datasets (e.g. `runners/simulations/simulate_train_test_data.py`), train a model (`deepdive_model_training.py`), and predict diversity from input file (e.g. `runners/empirical_analyses/predict_marine.py`). Run time for simulations, training and predictions will depend on the size of the dataset, complexity of the model and CPU. For our datasets they can be expected to last from several hours to a couple days, depending on the number of CPUs available. The runners include examples to parse the output data and plot them. 

We also include an R script (`setup_empirical_inputfiles/data_pipeline_utilities.R`) providing utility functions to pre-process fossil occurrence data and prepare input files that can be parsed by DeepDive. The script relies on the following R libraries:

```
dplyr  
readxl  
data.table  
tidyr  
stringr  
divdyn  
Rcpp  
```

which can be installed via CRAN. the script was tested on R v.4.1.2. We provide example files demonstrating the use of the script (e.g. `data_pipeline_marine.R`). 


The R script and runners can be used to reproduce all results shown in the paper. 


