# COTA
This project is a PyTorch implementation of COTA which is submitted to the PAKDD 2024.

## Prerequisties 
The implementation is based on Python 3.8.18 and PyTorch 1.6.0
A complete list of required packages can be found in the `requirements.txt` file.
Please install the necessary packages before running the code.

## Datasets
We use 3 datasets in our work: Amazon Beauty, Amazon Books, and Steam.
The preprocessed dataset is included in the repository: `./data`.


## Running the code
To execute the code, use the command `python main.py` with the arguments `--dataset` and `--train_dir`.
+ dataset
  + beauty
  + books
  + steam
 
```
python main.py --dataset=beauty --train_dir=default
```

For convenience, we provide a `demo.sh` script that reproduces the experiments presented in our work.
