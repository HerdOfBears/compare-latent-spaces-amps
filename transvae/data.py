import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *

def vae_data_gen(data, max_len=126, name=None, props=None, char_dict=None):
    """
    Encodes input smiles to tensors with token ids

    Arguments:
        mols (np.array, req): Array containing input molecular structures
        props (np.array, req): Array containing scalar chemical property values
        char_dict (dict, req): Dictionary mapping tokens to integer id
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    """
    seq_list = data[:,0] #unpackage the smiles: mols is a list of lists of smiles (lists of characters) 
    if props is None:
        props = np.zeros(seq_list.shape)
        props = props.reshape(-1,1)
        n_prop_outputs = 1
    else:
        # props = props.astype(int)
        n_props = len(props)
        n_seqs  = len(seq_list)
        n_prop_outputs = props.shape[1]
        if len(props) < len(seq_list):
            _extender = np.array([np.nan]*((n_seqs-n_props)*n_prop_outputs)).reshape(-1,n_prop_outputs)
            props = np.concatenate((props, _extender), axis=0)
    del data
    condn1 = (not name == None)
    condn2 = ("peptide" in name)
    if condn1 and condn2:  #separate sequence into list of chars e.g. 'CC1c2'-->['C''C''1''c''2']
        seq_list = [peptide_tokenizer(x) for x in seq_list]     #use peptide_tokenizer                  
    else: 
        seq_list = [tokenizer(x) for x in seq_list] 
    encoded_data = torch.empty((len(seq_list), max_len+1 + n_prop_outputs )) #empty tensor: (length of entire input data, max_seq_len + 2->{start & end})
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
