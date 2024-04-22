import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


import pandas as pd
import numpy as np
import torch
import os
import pickle as pkl

from transvae import trans_models
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN
from transvae.tvae_util import *
from transvae import analysis
from scripts.parsers import model_init, train_parser

from sklearn.decomposition import PCA
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# from ..oracles.AMPlify.src.AMPlify import Amplify

def encode_seqs(df, stoi):
    encoded_seqs = []
    max_len = 101
    for seq in df['peptides']:
        temp_ = [stoi[aa] for aa in seq]
        seq_len = len(seq)
        temp_ += [stoi["<end>"]]
        temp_ += [stoi["_"]] * (max_len - seq_len)
        temp_ = [0] + temp_
        encoded_seqs.append(temp_)
    df = pd.DataFrame({"encoded_peptides":encoded_seqs})
    return df
def decode_seq(encoded_seq, stoi):
    itos = {v:k for k,v in stoi.items()}
    decoded_seq = []
    for tok in encoded_seq:
        decoded_seq.append(itos[tok])
    decoded_seq = "".join(decoded_seq)
    decoded_seq = decoded_seq.strip("_")
    decoded_seq = decoded_seq.strip("<start>")
    decoded_seq = decoded_seq.strip("<end>")
    return decoded_seq


class OptimizeInReducedLatentSpace():
    def __init__(self, 
                 generative_model, 
                 property_oracle, 
                 dimensionality_reduction,
                 char_dict 
        ):
        """
        Class to perform Bayesian Optimization in the reduced latent space of a generative model.

        Parameters
        ----------
        generative_model : torch.nn.Module-like
            A trained generative model that can be used to generate sequences.
            Should have a method `greedy_decode` that can decode a latent space vector to a sequence.
        property_oracle :  
            Supervised learning model that predicts a property of a sequence.
            Must have a method `predict` that takes a list of sequences and 
            returns a list of predictions and confidence scores.
        dimensionality_reduction : 
            A dimensionality reduction model. Similar to sklearn's PCA.
            Must have a method `transform` and `inverse_transform`.
        char_dict : dict
            A dictionary that maps sequence characters to integers.
        """
        
        # Generative Model and Property Oracle setup
        self.generative_model = generative_model
        self.char_dict = char_dict

        self.property_oracle = property_oracle

        # Dimensionality Reduction setup
        # self.n_components = n_dim_components
        self.dimensionality_reducer = dimensionality_reduction
        self.n_reduced_dims = self.dimensionality_reducer.n_components

        self.optimization_results = {
            "iterations":[],
            "candidates":[],
            "candidate_classes":[],
            "candidate_scores":[]
        }

    def decode_seq(self, encoded_seq:list[int]) -> str:
        """
        Decodes an encoded sequence to a list of characters.

        Parameters
        ----------
        encoded_seq : list[int]
            A list of integers representing a sequence.

        Returns
        -------
        str
            The string representation of the sequence.
        """
        itos = {v:k for k,v in self.char_dict.items()}
        output = "".join([itos[i] for i in encoded_seq])
        output = output.strip("_")
        output = output.strip("<start>")
        output = output.strip("<end>")
        return output

    def encode_seq(self, sequence:str) -> list[int]:
        """
        Encodes a sequence to a list of integers.

        Parameters
        ----------
        sequence : str
            A sequence of characters.

        Returns
        -------
        list[int]
            A list of integers representing the sequence.
        """
        stoi = self.char_dict
        output = [stoi[i] for i in sequence]
        return output
        
    def get_fitted_gp_model(self, train_X, train_Y):
        """
        Fits the Gaussian Process model to the given data.

        Parameters
        ----------
        train_X : torch.Tensor
            The input data.
        train_Y : torch.Tensor
            The output data.
        """
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def optimize(self, train_X, train_Y, n_iters=10):

        train_X=train_X.to(float)
        train_Y=train_Y.to(float)

        char_dict = self.char_dict
        
        self.bounds = torch.stack([torch.ones(self.n_reduced_dims)*(-10), torch.ones(self.n_reduced_dims)*10])
        
        if self.optimization_results["iterations"]:
            last_iteration = self.optimization_results["iterations"][-1]
        else:
            last_iteration = 0
        print("============================================\nStarting optimization...")
        for i in range(n_iters):

            # fit and get the GP model
            self.gp = self.get_fitted_gp_model(train_X, train_Y)
            self.UCB = UpperConfidenceBound(self.gp, beta=0.1)

            # grab candidate(s)
            candidate, acq_value = optimize_acqf(
                self.UCB, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
            )

            # inverse projection
            candidate_invproj = self.dimensionality_reducer.inverse_transform(candidate)
            candidate_invproj = torch.from_numpy(candidate_invproj)

            # decode the candidate sequence from the latent space to the original sequence space
            with torch.no_grad():
                candidate_decoded = self.generative_model.greedy_decode(candidate_invproj)

            candidate_sequence = self.decode_seq(candidate_decoded.flatten().numpy())

            print(f"candiate sequence: {candidate_sequence}")
            prediction_class, prediction_score = self.property_oracle.predict(
                [candidate_sequence]
            )
            class_int = 1 if prediction_class[0] == "AMP" else 0

            # update the training data for the GP model
            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor(class_int).float().reshape(1,1)], dim=0)

            print(f"iteration {i+1} completed. Prediction class: {prediction_class}, Prediction score: {prediction_score}")
            self.optimization_results["iterations"].append(i+1+last_iteration)
            self.optimization_results["candidates"].append(candidate_sequence)
            self.optimization_results["candidate_classes"].append(prediction_class)
            self.optimization_results["candidate_scores"].append(prediction_score)

        print("Optimization complete")


if __name__ == "__main__":

    #########################################
    # load some training data
    print("loading data...")
    data_fpath = "data/"
    train_seqs = pd.read_csv(data_fpath+"peptides_2024_train.txt")
    train_fctn = pd.read_csv(data_fpath+"peptides_2024_train_function.txt")
    with open(data_fpath+"char_dict_peptides_2024.pkl", 'rb') as f:
        char_dict = pkl.load(f)

    df = encode_seqs(train_seqs, char_dict)    
    df["amp_or_not"] = train_fctn['amp']

    n_amps     = 100
    n_non_amps = 100
    some_amps     = df[df.amp_or_not==1].sample(    n_amps)
    some_non_amps = df[df.amp_or_not==0].sample(n_non_amps)
    
    sampled_peptides = []
    sampled_fctns = []
    for i,seq in enumerate(some_amps["encoded_peptides"]):
        sampled_peptides.append( [decode_seq(seq,char_dict)] )
        sampled_fctns.append( [some_amps.iloc[i,1]])
    for i,seq in enumerate(some_non_amps["encoded_peptides"]):
        sampled_peptides.append( [decode_seq(seq,char_dict)] )
        sampled_fctns.append( [some_non_amps.iloc[i,1]])

    sampled_peptides = np.array(sampled_peptides)
    sampled_fctns = np.array(sampled_fctns)


    #########################################
    # load a trained generative model
    print("loading generative model...")
    model_src = "checkpointz/amp_rnn_organized/070_rnn-128_peptides_2024.ckpt"
    model_obj=torch.load(model_src, map_location=torch.device("cpu"))
    # model = TransVAE(load_fn=model_src, workaround="cpu")
    model = RNN(load_fn=model_src, workaround="cpu")
    model.params['HARDWARE']= 'cpu'

    model.params["BATCH_SIZE"] = n_amps + n_non_amps
    t0 = time.time()
    with torch.no_grad():
        z, mu, logvar = model.calc_mems(sampled_peptides, log=False,save=False)
        decoded_seqs  = model.reconstruct(np.c_[sampled_peptides,sampled_fctns],log=False,return_mems=False)
    print(f"time elapsed = {round(time.time()-t0,5)}s")

    #########################################
    # build a dimensionality reduction method
    print("building PCA...")
    pca = PCA(n_components=5)
    pca.fit(mu)
    pca_mu = pca.transform(mu)

    #########################################
    # load a trained property predictor/oracle
    # print("loading amplify...")
    # amplify = Amplify("oracles/AMPlify/models/", "balanced")

    #########################################
    # initialize the optimizer
    print("initializing optimizer...")
    train_X = torch.from_numpy(pca_mu.copy())
    train_Y = pd.concat([some_amps, some_non_amps], axis=0, ignore_index=True)
    train_Y = torch.from_numpy(train_Y[["amp_or_not"]].values)

    # optimizer = OptimizeInReducedLatentSpace(
    #     model, amplify, pca, char_dict
    # )

    # #########################################
    # # perform optimization
    # print("optimizing...")
    # optimizer.optimize(train_X, train_Y, n_iters=10)