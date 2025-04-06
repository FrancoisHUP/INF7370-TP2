# INF7370-TP2

Paper -> [HUPF10049509-INF7370-TP2](https://www.overleaf.com/read/sqkpkhfbcvqx#67323d)

## Environment Setup

You need to have Python 3.12.x or lower installed. TensorFlow is not supported on newer Python versions. To check your Python version, use this command:

```bash
python --version
```

If your version is above 3.12.x, follow these instructions; otherwise, you can skip these steps.

### Install Python

**Windows**

To use a 3.12 environment, you can either use Conda or [download Python version 3.12](https://www.python.org/downloads/release/python-3129/) and run:

```bash
/c/Users/<user_name>/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv &&
source .venv/Scripts/activate && python --version
```

> **NOTE:** Make sure to replace `<user_name>` with your actual username.

**Linux**

```bash
sudo apt install python3.12 python3.12-venv python3.12-dev -y
python3.12 -m venv .venv && source .venv/bin/activate && python --version
```

You should see:

```bash
Python 3.12.9
```

Then, install the required packages with:

```bash
pip install -r requirements.txt
```

### Download Data

We use a dataset of 30,000 images. You can download the [zip file](https://drive.google.com/drive/folders/1x6_nLO4wFT_nHMThUeeeb2Hj-TcE84Bi?usp=sharing) with the following commands:

```bash
pip install -q gdown
gdown --id 18-fICo3gs8LuwqtPp17VFdBNge5q-58M
```

## Model, Training, and Evaluation

You can build the model and start training with this command:

```bash
python 1_Modele.py
```

Once the training is done, you can see the results using:

```bash
tensorboard --logdir logs/
```

You can resume training if, for some reason, it stops by using:

```bash
python 1_Modele.py --resume models/default_epoch31.keras --epochs 50
```

## Evaluation

To evaluate the model, use the following command:

```bash
python 2_Evaluation.py
```

> Using a GPU can be hard to set up. Here is a good tutorial that might help you:  
> [How to Resolve “cuDNN, cuFFT, and cuBLAS Errors” on CentOS 7](https://funnymove.medium.com/how-to-resolve-cudnn-cufft-and-cublas-errors-on-centos-7-7958f00a6d0d).