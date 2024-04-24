import numpy as np
import torch
import pandas as pd
import MDAnalysis as mda

import logging
import os

from transformers import AutoTokenizer, EsmForProteinFolding

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

    df = pd.read_csv("data/peptides_2024_train.txt")
    df = df.sample(10)
    
    sequences = df["peptides"].tolist()

    model = EsmForProteinFolding.from_pretrained(
                        model_path,
                        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True
    )
    print("model and tokenizer loaded")

    # inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    print('inference')
    x_structures_subset = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            # outputs = model.infer_pdb(seq)
            outputs = model.infer(seq)
            x_structures_subset.append(outputs)

    print("=====================================")
    print(x_structures_subset[0].keys())
    print("=====================================")

    print(f"computing rmsds")
    _rmsds = []
    for i in range(len(x_structures_subset)):
        print(f"comparing {i} to other structures")
        for j in range(i+1, len(x_structures_subset)):
            _uni1 = mda.Universe(x_structures_subset[i].positions)
            _uni2 = mda.Universe(x_structures_subset[j].positions)
            _rmsds.append(
                mda.analysis.rms.rmsd(
                    _uni1.select_atoms("backbone"),
                    _uni2.select_atoms("backbone")
                )
            )
    
    print(_rmsds)