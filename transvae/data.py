import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time, logging
from torch.autograd import Variable

from transvae.tvae_util import *

def vae_data_gen(data, max_len=126, name=None, props=None, char_dict=None, d_pp_out=1, mask_label_percent=None, mask_rng_seed=42):
    """
    Encodes input smiles to tensors with token ids

    Arguments:
        mols (np.array, req): Array containing input molecular structures
        props (np.array, req): Array containing scalar chemical property values
        char_dict (dict, req): Dictionary mapping tokens to integer id
        d_pp_out (int, opt): Number of property outputs. Default is 1
        mask_label_percent: percentage of labels to mask. Default is None
        mask_rng_seed: random seed for masking. Default is 42
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    Notes:
        - If props is None, a tensor of zeros is created
        - when using masked labels, the percentage to mask corresponds to the percentage of the full dataset,
            not the percentage of the batch, nor the percentage of the non-NaN labels. 
            E.g. even if only 10% of the full dataset is labelled, if mask_label_percent=0.1, then a random 10% of the full dataset will be masked.
    """
    seq_list = data[:,0] #unpackage the smiles: mols is a list of lists of smiles (lists of characters) 
    if props is None:
        # props = np.zeros(seq_list.shape)
        props = np.zeros( (len(seq_list), d_pp_out) )
        n_prop_outputs = d_pp_out
    else:
        # props = props.astype(int)
        n_props = len(props)
        n_seqs  = len(seq_list)
        n_prop_outputs = props.shape[1]
        if len(props) < len(seq_list):
            _extender = np.array([np.nan]*((n_seqs-n_props)*n_prop_outputs)).reshape(-1,n_prop_outputs)
            props = np.concatenate((props, _extender), axis=0)
    del data

    if mask_label_percent is not None:
        _generator = np.random.default_rng(mask_rng_seed)
        mask_labels = _generator.choice([True, False], size=props.shape, p=[mask_label_percent, 1-mask_label_percent])
        props[mask_labels] = np.nan

    logging.info(f"tokenizing sequences")
    condn1 = (not name == None)
    condn2 = ("peptide" in name)
    if condn1 and condn2:  #separate sequence into list of chars e.g. 'CC1c2'-->['C''C''1''c''2']
        seq_list = [peptide_tokenizer(x) for x in seq_list]     #use peptide_tokenizer                  
    else: 
        seq_list = [tokenizer(x) for x in seq_list] 
    encoded_data = torch.empty((len(seq_list), max_len+1 + n_prop_outputs )) #empty tensor: (length of entire input data, max_seq_len + 2->{start & end})
    logging.info(f"Encoding sequences with start tokens and padding")
    for j, seq in enumerate(seq_list):
        encoded_seq = encode_seq(seq, max_len, char_dict) #encode_smiles(smile,max_len,char_dict): char dict has format {"char":"number"}
        encoded_seq = [0] + encoded_seq
        encoded_data[j,               :-n_prop_outputs] = torch.tensor(encoded_seq)
        encoded_data[j,-n_prop_outputs:               ] = torch.tensor(props[j,:])
    return encoded_data

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
