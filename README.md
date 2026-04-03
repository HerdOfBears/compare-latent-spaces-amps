# Compare Latent Spaces for AMP design
This repo contains the code used for the paper [*Towards best practices in low-dimensional semi-supervised latent Bayesian optimization for the design of antimicrobial peptides*](https://arxiv.org/abs/2510.17569) to be published in MSDE. The base neural network architectures were implemented and tested in a previous work by [Renaud and Mansbach (2023)](https://doi.org/10.1039/D2DD00091A). 

In this work we used Bayesian Optimization with Gaussian Processes over a VAE's latent space to design antimicrobial peptides. We compared latent spaces organized by easily-computed peptide properties, similar to [Gomez-Bombarelli et al. (2018)](https://pubs.acs.org/doi/pdf/10.1021/acscentsci.7b00572), or by the oracle itself. We expanded on Gomez-Bombarelli et al.'s line of work by (1) comparing latent spaces organized with varying amounts of property labels, (2) analyzing optimization in a low-dimensional linear projection (PCA) of the high-dimensional latent space, (3) performing a preliminary comparison to [deep kernel learning](https://proceedings.mlr.press/v51/wilson16.html?ref=https://githubhelp.com). 

Compared to [latent-spaces-amps](https://github.com/Mansbach-Lab/latent-spaces-amps), the primary new files related to BayesOpt in latent spaces are:
- transvae/optimization.py
-- This implements the Bayesian Optimization loop. It assumes you provide a generative model with a greedy\_decode method, a dimensionality\_reduction object with transform and inverse\_transform methods, and an oracle that will score new sequences with a predict method.
- transvae/deep_kernel_learning.py
-- This implements a simple neural network 'FeatureExtractor' class, a GPyTorch kernel using the neural net, and a fitting function to fit the neural network and base kernel's parameters using backpropagation. 
- scripts/bayesian_optimization_loop.py
-- This script is the main runner script for spawning BayesOpt loop processes for our experiments.

notebooks/ contains jupyter notebooks used to produce figures for the manuscript.
model checkpoints and the datasets used for training models (and fitting the SVR) are available at a [zenodo deposit](doi.org/10.5281/zenodo.17872449)

# Installation
Installing the transvae package with the Bayesian optimization additions requires cloning this repo to your device. Then setting up either a python virtual environment or conda environment. This package depends primarily on BoTorch, PyTorch, scikit-learn, gpytorch. 