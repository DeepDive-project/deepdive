# DeepDive 

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
