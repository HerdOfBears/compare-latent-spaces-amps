
import comet_ml
import os
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM


# Function to compute sequence likelihoods
def compute_likelihoods(sequences, model, tokenizer, device, chunk_size=1000,test="OFF"):
	model.eval()
	likelihoods = []
	n_sequences = len(sequences)
	loglikelihoods = torch.zeros(n_sequences)
	
	N_CHUNKS = int(math.ceil(len(sequences)/chunk_size))

	if test=="ON":
		N_CHUNKS = 1
		n_sequences = chunk_size
		loglikelihoods = torch.zeros(n_sequences)

	t0 = time.time()
	# for i, seq in enumerate(sequences[:n_sequences-1000]):
	print(f"{N_CHUNKS=} in compute_likelihoods")
	for i in range(N_CHUNKS):
		if i%10==0:
			tf = time.time()
			print(f"processed {i}/{N_CHUNKS}, taking {tf-t0} for 1000")
			t0= time.time()
		inputs = tokenizer(sequences[i*chunk_size:(i+1)*chunk_size], return_tensors='pt',padding=True).to(device)
		with torch.no_grad():
			outputs = model(**inputs)
			log_likelihood = F.log_softmax(outputs.logits, dim=-1)
			input_ids = inputs.input_ids
			seq_likelihood = log_likelihood.gather(2, input_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)
			
			loglikelihoods[i*chunk_size:(i+1)*chunk_size] = seq_likelihood.cpu()
			# likelihoods.append(seq_likelihood.item())
	return loglikelihoods
  
def compute_representations(sequences, model, tokenizer, device, chunk_size=1000,test="OFF"):
	model.eval()
	outputs = torch.zeros(len(sequences), model.config.hidden_size)
	N_CHUNKS = int(math.ceil(len(sequences)/chunk_size))
	
	if test=="ON":
		N_CHUNKS = 1
		n_sequences = chunk_size
		outputs = torch.zeros(n_sequences)

	t0 = time.time()
	with torch.no_grad():
		for _chunk_idx in range(N_CHUNKS):
			if _chunk_idx%10==0:
				tf = time.time()
				print(f"on chunk {_chunk_idx}/{N_CHUNKS}, taking {tf-t0} ")
				t0 = time.time()

			_chunk = sequences[_chunk_idx*chunk_size:(_chunk_idx+1)*chunk_size]
			
			_inputs = tokenizer(_chunk, return_tensors='pt',padding=True).to(device)
			
			repres = model(**_inputs).last_hidden_state[:,0,:] # take <cls> token to obtain sequence-level representation

			# first token <cls> corresponds to sequence level representation
			outputs[_chunk_idx*chunk_size:(_chunk_idx+1)*chunk_size,:] = repres.cpu()
	
	return outputs


if __name__ == "__main__":
	
	# parse cmd-line arguments
	parser = argparse.ArgumentParser()

	# add arguments, including model dir, data file path, comet api key, and project name
	parser.add_argument("--model_dir",  type=str, required=True)
	parser.add_argument("--data_fpath", type=str, required=True)
	parser.add_argument("--output_dir", type=str, required=True)

	parser.add_argument("--output_name", type=str, required=False, default="cdhit90_semibalanced")
	parser.add_argument("--comet", type=str, required=False, default="OFF")	
	parser.add_argument("--comet_api_key", type=str, required=False)
	parser.add_argument("--comet_project_name", type=str, required=False)
	parser.add_argument("--test", type=str, required=False, default="OFF")
	# parse the arguments
	args = parser.parse_args()

	# Set the model directory and data file path
	model_dir  = args.model_dir
	data_fpath = args.data_fpath
	output_dir = args.output_dir
	TEST=args.test

	if args.comet == "ON":
		comet_api_key = args.comet_api_key
		comet_project_name = args.comet_project_name
		
		experiment = comet_ml.Experiment(
			api_key=comet_api_key, 
			project_name=comet_project_name
		)

	# Load the model
	esm_model = EsmForMaskedLM.from_pretrained(
		model_dir, 
		local_files_only=True
	)
	esm_tokenizer = EsmTokenizer.from_pretrained(
		model_dir, 
		local_files_only=True
	)

	# Load the sequences
	sequences = pd.read_csv(data_fpath)['sequence'].values

	# Set the device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Compute the likelihoods
	print("Computing likelihoods...")
	likelihoods = compute_likelihoods(
		sequences, 
		esm_model, 
		esm_tokenizer, 
		device,
		chunk_size=2000,
		test=TEST
	)


	# Compute the representations
	print("Computing representations...")
	representations = compute_representations(
		sequences,
		esm_model.esm,
		esm_tokenizer,
		device,
		chunk_size=2000,
		test=TEST
	)

	# Save the likelihoods and representations
	print("Saving likelihoods and representations...")
	np.save(f'{output_dir}/loglikelihoods-esm2_650m.npy', likelihoods)
	np.save(f'{output_dir}/representations-esm2_650m.npy', representations)
	print("Done!")