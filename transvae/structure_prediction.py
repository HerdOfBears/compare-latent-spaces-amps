import numpy as np
import torch
import pandas as pd
import MDAnalysis as mda
import Bio.PDB
from Bio.PDB import PDBParser, cealign

import logging
import os

from transformers import AutoTokenizer, EsmForProteinFolding

def batch_sequence_to_pdb(sequences:list[str], model_path:str) -> list[str]:
    """
    Using ESMFold, takes a list of amino acid sequences and returns a list of strings, 
    where each string is the content of a PDB file representing the predicted structure 
    of the input sequence.

    Parameters
    ----------
    sequences : list[str]
        list of sequences to predict the structure of
    model_path : str
        path to the ESMFold model weights

    Returns
    -------
    structures_pdbs: list[str]
        each string is the content of a PDB file representing the 
        predicted structure
    """

    if not os.path.isdir( os.path.join(os.getcwd(),model_path) ):
        raise Exception("""Model path does not exist: {}\n
                        current directory is: {}""".format(model_path, os.getcwd())
        )
    if not isinstance(sequences, list):
        sequences = list(sequences)
        
    logging.info("Loading model...")
    model = EsmForProteinFolding.from_pretrained(
                        model_path,
                        local_files_only=True
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #                     model_path,
    #                     local_files_only=True
    # )

    structures_pdbs = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            outputs = model.infer_pdb(seq)
            structures_pdbs.append(outputs)

    return structures_pdbs

def pdb_to_biostructure(pdb_file_contents:str|list) -> list[Bio.PDB.Structure]:
    """
    takes a string or list of strings containing the contents of a PDB file. 
    Returns a list of BioPython Bio.PDB.Structure objects representing the
    input PDB files.
    In particular, this assumes that the input PDB file contains a single string with newline characters.

    Parameters
    ----------
    pdb_file_contents : str or list[str]
        string or list of strings containing the contents of a PDB file(s)
    
    Returns
    -------
    structures : list[Bio.PDB.Structure]
        list of BioPython Bio.PDB.Structure objects representing the input PDB files
    """
    parser = Bio.PDB.PDBParser(QUIET=True)

    _inputs = None
    if isinstance(pdb_file_contents, str):
        _inputs = [pdb_file_contents]
    else:
        _inputs = pdb_file_contents

    structures = []
    for i, _pdb in enumerate(_inputs):
        parser.structure_builder.init_structure(f"structure{i}")

        parser._parse(_pdb.split("\n"))

        parser.structure_builder.set_header(parser.header)

        # Return the Structure instance
        _structure = parser.structure_builder.get_structure()

        structures.append(_structure)

    return structures

def biostructure_to_rmsds(biostructures:list[Bio.PDB.Structure])->np.ndarray:
    """
    takes a list of BioPyton Bio.PDB.Structure objects and
    computes the pairwise RMSD values between them

    Parameters
    ----------
    biostructures : list[Bio.PDB.Structure], length N
        list of BioPython Bio.PDB.Structure objects

    Returns
    -------
    rmsds : np.ndarray shape (N*(N-1)/2, 1)
        pairwise RMSD values between the structures
    """
    
    aligner = cealign.CEAligner()
    
    rmsds = []
    N = len(biostructures)
    for i in range(N):
        aligner.set_reference(biostructures[i])
        for j in range(i+1, N):
            aligner.align(biostructures[j])

            rmsds.append(
                aligner.rms
            )
    rmsds = np.array(rmsds).reshape(-1,1)
    return rmsds


if __name__ == "__main__":

    model_path = "./esmfold_v1/"

    df = pd.read_csv("data/peptides_2024_train.txt")
    df = df.sample(10)
    
    sequences = df["peptides"].tolist()

    print("predicting structures...")
    structures_pdbs = batch_sequence_to_pdb(sequences, model_path)
    print("done structure prediction")

    print("converting to BioPython structures...")
    structures = pdb_to_biostructure(structures_pdbs)
    print("done BioPython conversion")

    print("computing RMSDs...")
    rmsds = biostructure_to_rmsds(structures)
    print("done RMSD computation")
    print(rmsds)