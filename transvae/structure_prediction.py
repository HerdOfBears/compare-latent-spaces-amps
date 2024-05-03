import numpy as np
import torch
import pandas as pd
import MDAnalysis as mda
import Bio.PDB
from Bio.PDB import PDBParser, cealign, ccealign # ccealign has one function: run_cealign(coordsA, coordsB, windowSize, gapMax)
from Bio.PDB.qcprot import QCPSuperimposer

import logging
import os
import time

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
    if N < 2:
        rmsds = np.zeros((1,1)) - 1 # rmsd can't be negative, so -1 indicates error
        return rmsds
    
    for i in range(N):
        aligner.set_reference(biostructures[i])
        for j in range(i+1, N): 
            # aligner.align(biostructures[j]) #FLAG
            
            ############################################
            # Run CEAlign
            # CEAlign returns the best N paths, where each path is a pair of lists
            # with aligned atom indices. Paths are not guaranteed to be unique.
            coord = aligner.get_guide_coord_from_structure(biostructures[j])
            paths = ccealign.run_cealign(aligner.refcoord, coord, aligner.window_size, aligner.max_gap)
            unique_paths = {(tuple(pA), tuple(pB)) for pA, pB in paths}

            # Iterate over unique paths and find the one that gives the lowest
            # corresponding RMSD. Use QCP to align the molecules.
            best_rmsd = 1e6
            for u_path in unique_paths:
                idxA, idxB = u_path

                coordsA = np.array([aligner.refcoord[i] for i in idxA])
                coordsB = np.array([coord[i] for i in idxB])

                aln = QCPSuperimposer()
                aln.set(coordsA, coordsB)
                # aln.run()
                aln.rms=1
                _temp = 1
                if _temp < best_rmsd:
                    best_rmsd = _temp

            ############################################
            # rmsds.append( # FLAG
            #     aligner.rms
            # )
    
    # construct array of zeros
    rmsds = np.zeros((N*(N-1)//2, 1)) #FLAG

    print("creating numpy array:")
    # rmsds = np.array(rmsds).reshape(-1,1)
    print("created array")
    return rmsds


class StructurePredictor:
    def __init__(self, model_path:str, device="cpu"):
        self.model_path = model_path
        self.model = EsmForProteinFolding.from_pretrained(
                        model_path,
                        local_files_only=True
        )
        if device in ["gpu", "cuda"]:
            print("Using GPU for structure prediction")
            self.model = self.model.cuda().requires_grad_(False)
        else:
            self.model = self.model.requires_grad_(False)

    def predict_structures(self, sequences:list[str]) -> list[str]:
        """
        Using ESMFold, takes a list of amino acid sequences and returns a list of strings, 
        where each string is the content of a PDB file representing the predicted structure 
        of the input sequence.

        Parameters
        ----------
        sequences : list[str]
            list of sequences to predict the structure of

        Returns
        -------
        structures_pdbs: list[str]
            each string is the content of a PDB file representing the 
            predicted structure
        """

        if not isinstance(sequences, list):
            sequences = list(sequences)
            
        logging.info("Loading model...")
        min_sequence_length = 16 # because of CEAlign window_size = 8
        structures_pdbs = []
        with torch.no_grad():
            for i, seq in enumerate(sequences):
                if len(seq)<min_sequence_length:
                    continue
                outputs = self.model.infer_pdb(seq)
                structures_pdbs.append(outputs)

        return structures_pdbs

    def pdb_to_biostructure(self, pdb_file_contents:str|list) -> list[Bio.PDB.Structure]:
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
        print("finished making structures")
        return structures


if __name__ == "__main__":

    model_path = "./esmfold_v1/"
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"
    print('using device:', device)
    model = StructurePredictor(model_path, device=device)

    df = pd.read_csv("data/peptides_2024_train.txt")
    df = df.sample(10)
    
    sequences = df["peptides"].tolist()

    print("predicting structures...")
    # structures_pdbs = batch_sequence_to_pdb(sequences, model_path)
    structures_pdbs = model.predict_structures(sequences)
    print("done structure prediction")

    print("converting to BioPython structures...")
    # structures = pdb_to_biostructure(structures_pdbs)
    structures = model.pdb_to_biostructure(structures_pdbs)
    print("done BioPython conversion")

    print("computing RMSDs...")
    rmsds = biostructure_to_rmsds(structures)
    print("done RMSD computation")
    print(rmsds)