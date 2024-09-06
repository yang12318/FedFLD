## Setup

To run the code, you need the following dependencies:

- torch==1.12.1

- numpy==1.21.0

- sklearn==1.2.2

You can simply run 

```python
pip install -r requirements.txt
```

## Download dataset

Create `"path/to/data"` directory under the root of this project.

Download datasets, and put them into `"path/to/data"`.

## Preprocess dataset

`prepare_data.py` preprocess datasets into FL settings.

## Run the script

Quick launch the experiment by running the script:

```
bash CIFAR10_FedFLD.sh
```