import pandas as pd
import numpy as np
import scipy.optimize as scop

import peptides
import propy

import matplotlib.pyplot as plt

import argparse
import pickle as pkl
import os
import time

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit

from sklearn.feature_selection import mutual_info_regression

from transvae.mic_svr import compute_propy_properties, VS_SSVR, perform_mRMR

def load_data(data_dir, data_type="mic"):

    # load data as used by Witten & Witten (2019)
    if data_type=="mic":
        propy_des = pd.read_csv(data_dir+"train_propy_des.csv")

        ecoli_train_with_c = pd.read_pickle(f'{data_dir}ecoli_train_with_c_df.pkl')
        train_Y = ecoli_train_with_c.value
    elif data_type=="hemolytik":
        propy_des = pd.read_csv(data_dir+"train_propy_des_hemolytik.csv")

        hemolytik_train = pd.read_csv(data_dir+"train_log10_HC50.csv")
        train_Y = hemolytik_train.log10_HC50
    else:
        raise ValueError("data_type must be either 'mic' or 'hemolytik'")
    

    input_data = propy_des.dropna()
    train_Y = train_Y[(propy_des.isna().sum(axis=1)==0).values]

    train_Y = train_Y.loc[
        ((input_data.sequence.str.len()>4) &
        (input_data.sequence.str.len()<101)).values
    ]

    input_data = input_data.loc[
        (input_data.sequence.str.len()>4) &
        (input_data.sequence.str.len()<101)
    ]
    input_data = input_data.loc[:,input_data.columns[1:]]

    return input_data, train_Y

def main(data_dir, N_CPUS=1, data_type="mic"):

    # gets training and testing data, as used by Witten & Witten (2019). 

    if os.getcwd().split("/")[-1] == "scripts":
        data_dir = "../oracles/"
    else:
        data_dir = "oracles/"

    input_data, train_Y = load_data(data_dir, data_type=data_type)

    
    t0 = time.time()
    results, relevancies, pairwise_redundancies = perform_mRMR(
        input_data, train_Y, N_CPUS=N_CPUS, max_features=input_data.shape[1], verbose=True
    )
    
    print(f"{time.time()-t0}s")

    print("pickling results")
    mrmr_results = {
        "results": results,
        "relevancies": relevancies,
        "pairwise_redundancies": pairwise_redundancies
    }
    with open(data_dir+f"mrmr_results_{data_type}.pkl", "wb") as f:
        pkl.dump(mrmr_results, f)
    print("done mRMR")


if __name__ == "__main__":
    # take data directory and number of cpus as command-line arguments
    parser = argparse.ArgumentParser(description="Run mRMR feature selection")
    parser.add_argument("--data_dir", type=str, help="path to the data directory")
    parser.add_argument("--n_cpus", type=int, help="number of cpus to use")
    parser.add_argument("--data_type", type=str, help="data to use (mic or hemolytik)")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    N_CPUS = args.n_cpus
    data_type = args.data_type

    # check that data_type is in ["mic", "hemolytik"]
    if data_type not in ["mic", "hemolytik"]:
        raise ValueError("data_type must be either 'mic' or 'hemolytik'")
    
    print(f"{data_dir=}, {N_CPUS=}, {data_type=}")

    main(data_dir, N_CPUS=N_CPUS, data_type=data_type)