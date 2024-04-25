import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch

#from transvae import *
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from transvae.structure_prediction import StructurePredictor
from scripts.parsers import model_init, train_parser

def train(args):
    ### Update beta init parameter from loaded chekpoint
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=torch.device('cuda'))
        start_epoch = ckpt['epoch']
        total_epochs = start_epoch + args.epochs
        beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
        args.beta_init = beta_init

    if 'ON' in args.DDP: #bizare behaviour of arg parser with booleans means we convert here...
        args.DDP= True
    else:
        args.DDP=False
    if 'ON' in args.property_predictor:
        args.property_predictor= True
    else: 
        args.property_predictor = False
    
    ### Build params dict from the parsed arguments
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'HARDWARE' : args.hardware,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps,
              'INIT_METHOD': args.init_method,
              'DIST_BACKEND': args.dist_backend,
              'WORLD_SIZE': args.world_size,
              'DISTRIBUTED': args.distributed,
              'NUM_WORKERS': args.num_workers,
              'DDP': args.DDP,
              'DISCRIMINATOR_LAYERS' : args.discriminator_layers,
              'LOSS_METHOD': args.loss_method,
              'd_pp_out': args.d_pp_out,
              'prediction_types': args.prediction_types,          
    }

    ### Load data, vocab and token weights
    train_mols = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
    test_mols  = pd.read_csv( 'data/{}_test.txt'.format(args.data_source)).to_numpy()
    if args.property_predictor:
        assert args.train_props_path is not None and args.test_props_path is not None, \
        "ERROR: Must specify files with train/test properties if training a property predictor"
        train_props = pd.read_csv(args.train_props_path).to_numpy()
        test_props  = pd.read_csv( args.test_props_path).to_numpy()
        
        if (args.prediction_types is None) or (set(args.prediction_types) == set(["classification"])):
            train_props = train_props.astype(int)
            test_props  =  test_props.astype(int)
        
        # raise error if number of properties in properties file does not match d_pp_out
        if train_props.shape[1] != args.d_pp_out:
            raise ValueError(f"Number of properties in properties file {train_props.shape[1]} does not match d_pp_out ({args.d_pp_out}) ")
    else:
        train_props = None
        test_props  = None
    with open('data/char_dict_{}.pkl'.format(args.data_source), 'rb') as f:
        char_dict = pickle.load(f)
    char_weights = np.load('data/char_weights_{}.npy'.format(args.data_source))
    params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params[ 'ORG_DICT'] =  org_dict

    ### if using structure loss, load structure predictor
    # args.structure_model_path
    # args.structure_loss
    if "yes" in args.use_structure_loss:
        print(f"Using structure model from {args.structure_model_path}")
        if args.hardware == "gpu":
            esmfold = StructurePredictor(args.structure_model_path, device="gpu")
        else:
            esmfold = StructurePredictor(args.structure_model_path)
        use_structure_loss = True
    else:
        esmfold = None
        use_structure_loss = False

    ####################
    ### Train model
    ####################
    vae = model_init(args, params)
    if args.checkpoint is not None:
        vae.load(args.checkpoint)
    vae.train(train_mols, test_mols, train_props, test_props,
              epochs=args.epochs, save_freq=args.save_freq,
              use_structure_loss=use_structure_loss, structure_predictor=esmfold
    )


if __name__ == '__main__':
    print("main function called /n")
    parser = train_parser()
    args = parser.parse_args()
    
    # quick fix of parser bug. parsing prediction_types as a list of strings rather than a list of characters
    if args.prediction_types is not None:
        fixed_prediction_types = []
        if isinstance(args.prediction_types[0],list):
            for l in args.prediction_types:
                fixed_prediction_types.append("".join(l))
        elif isinstance(args.prediction_types[0],str):
            fixed_prediction_types = ["".join(args.prediction_types)]
        else:
            raise ValueError("prediction_types must be a list of strings or a list of lists of characters")
        args.prediction_types = fixed_prediction_types
    train(args)
