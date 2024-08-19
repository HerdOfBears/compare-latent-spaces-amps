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

def lora_fine_tune_esm2(sequences, targets,model, tokenizer, epochs=3, batch_size=16,device=None):

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
    peft_model = get_peft_model(model, lora_config).to(device)

    # check if it's working
    peft_model.print_trainable_parameters()

    # Optimizer and scheduler setup
    optimizer = AdamW(peft_model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=epochs * len(sequences) // batch_size
    )

    # Tokenize the sequences
    inputs = tokenizer(sequences.tolist(), return_tensors='pt', padding=True, truncation=True).to(device)
        
    labels = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1).to(device)

    peft_model.train()
    for epoch in range(epochs):
        for i in range(0, len(sequences), batch_size):
            batch_inputs = {key: val[i:i + batch_size].to(device) for key, val in inputs.items()}
            batch_labels = labels[i:i + batch_size].to(device)
            
            outputs = peft_model(**batch_inputs, labels=batch_labels)
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return peft_model, lora_config


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True,
                        description="Number of labels for the classification. For regression, set to 1.")

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
                                     device=device)


    # Save the fine-tuned model
    peft_model.save_pretrained( args.output_path)
    peft_config_dict = peft_config.to_dict()
    with open(f"{args.output_path}/peft_config_test.pkl", "wb") as f:
        pkl.dump(peft_config_dict, f)