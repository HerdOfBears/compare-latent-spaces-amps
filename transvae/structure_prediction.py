from transformers import AutoTokenizer, EsmForProteinFolding
import numpy as np
import torch

import logging
import os

import MDAnalysis as mda

def sequence_to_pdb_esm(sequence:str, model_path:str=None):
    # model_path = "./transvae/esmfold_v1/"
    if not os.path.isdir( os.path.join(os.getcwd(),model_path) ):
        raise Exception("""Model path does not exist: {}\n
                        current directory is: {}""".format(model_path, os.getcwd())
        )

    logging.info("Loading model...")
    model = EsmForProteinFolding.from_pretrained(
                        model_path,
                        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True
    )

    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        outputs = model.infer_pdb(sequence)

    return outputs

if __name__ == "__main__":

    model_path = "./esmfold_v1/"

    x_structures_subset = []
    sequences = []
    for i, seq in enumerate(sequences):
        outputs = sequence_to_pdb_esm(seq, model_path=model_path)
        x_structures_subset.append(outputs)

    _rmsds = []
    for i in range(len(x_structures_subset)):
        print(f"comparing {i} to other structures")
        for j in range(i+1, len(x_structures_subset)):
            _uni1 = mda.Universe(x_structures_subset[i])
            _uni2 = mda.Universe(x_structures_subset[j])
            _rmsds.append(
                mda.analysis.rms.rmsd(
                    _uni1.select_atoms("backbone"),
                    _uni2.select_atoms("backbone")
                )
            )
    
    print(_rmsds)