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
