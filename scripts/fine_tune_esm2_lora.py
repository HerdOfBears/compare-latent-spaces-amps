import argparse
import time
import logging
import math
import pickle as pkl

import torch
import pandas as pd
import numpy as np

from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from peft import PeftModel, LoraConfig
from peft import get_peft_config, get_peft_model

def peft_save(peft_model, peft_config, output_path, epoch):
    # Save the fine-tuned model
    _path_parts = output_path.split("/")
    if output_path[-1] == "/":
        _idx = -2
    else:
        _idx = -1
    _last_name=f"{epoch}_" + _path_parts[_idx]

    if len(_path_parts) == 1:
        output_path = _last_name + "/"
    else:
        output_path = "/".join(output_path.split("/")[:-1]) + "/" + _last_name + "/"

    peft_model.save_pretrained( output_path)
    peft_config_dict = peft_config.to_dict()
    with open(f"{output_path}/{epoch}_peft_config_test.pkl", "wb") as f:
        pkl.dump(peft_config_dict, f)

def lora_fine_tune_esm2(sequences, targets, peft_model, tokenizer, epochs=3, batch_size=16,device=None, params={}):

    # Initialize LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the adaptation matrix
        lora_alpha=32,  # Scaling factor
        target_modules=["query", "key", "value"],  # Targeting specific layers for adaptation
        lora_dropout=0.1  # Dropout rate
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply LoRA to the model
    peft_model = get_peft_model(peft_model, lora_config).to(device)

    # check if it's working
    peft_model.print_trainable_parameters()

    print("setting up adamW")
    # Optimizer and scheduler setup
    optimizer = AdamW(peft_model.parameters(), lr=5e-5)
    print("setting up scehduler")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=epochs * len(sequences) // batch_size
    )

    # Tokenize the sequences
    print(f"tokenizing...")
    inputs = tokenizer(sequences.tolist(), return_tensors='pt', padding=True, truncation=True).to(device)

    labels = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

    print(f"training loop")
    peft_model.train()
    for epoch in range(epochs):
        t0 = time.time()
        for i in range(0, len(sequences), batch_size):
            batch_inputs = {key: val[i:i + batch_size].to(device) for key, val in inputs.items()}
            batch_labels = labels[i:i + batch_size].to(device)

            outputs = peft_model(**batch_inputs, labels=batch_labels)

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        tf = time.time()
        print(f"Epoch {epoch+1}/{epochs}, loss {loss.item()}, time elapsed {tf-t0}s")

        if params.save_freq is not None:
            if (epoch+1) % params.save_freq == 0:
                print(f"saving model at {epoch+1}")
                peft_save(peft_model, lora_config, params.output_path, epoch+1)


    return peft_model, lora_config

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True,
                        help="Number of labels for the classification. For regression, set to 1.")
    
    # saving args
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save model every SAVE_FREQ epochs")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ESM-2 model and tokenizer
    model_path = args.model_path
    model = EsmForSequenceClassification.from_pretrained(
        model_path,
        num_labels=args.num_labels).to(device)
    
    tokenizer = EsmTokenizer.from_pretrained(
        model_path
    )

    # Load the dataset 
    data = pd.read_csv(args.data)

    sequences = data.iloc[:,0].to_numpy()
    targets   = data.iloc[:,1].to_numpy()

    # Fine-tune the ESM-2 model using LoRA
    peft_model, peft_config = lora_fine_tune_esm2(sequences, 
                                     targets, 
                                     model, 
                                     tokenizer,
                                     epochs=args.epochs,
                                     batch_size=args.batch_size,
                                     device=device,
                                     params=args)


    # Save the fine-tuned model
    peft_model.save_pretrained( args.output_path)
    peft_config_dict = peft_config.to_dict()
    with open(f"{args.output_path}/peft_config_test.pkl", "wb") as f:
        pkl.dump(peft_config_dict, f)