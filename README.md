# INF7370-TP2

## Setup environnement 

Tensorflow is now supported in newer version of python. Make sure you use a python environnement <=3.12.

**Windows**
 
To use a 3.12 env you can either use conda or [download python versino 3.12](https://www.python.org/downloads/release/python-3129/) and do : 

```bash
/c/Users/<user_name>/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv && 
source .venv/Scripts/activate && python --version
```

>**NOTE:** Make sure to change the <user_name> to you user. 

You should be this

```bash
Python 3.12.9 
```

**Linux**
```bash
sudo apt install python3.12 python3.12-venv python3.12-dev -y
python3.12 -m venv venv && source venv/bin/activate && python --version
```

## Training & model 

You can build the model and start the traning with this command  

```bash
python 1_Modele.py
```

Once the traning is done you can see the results using this command  

```bash
tensorboard --logdir logs/
```

## Evaluation 

To evalutate the model, use this command line : 

```bash
python 2_Evaluation.py
```

