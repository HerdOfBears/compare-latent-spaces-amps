import torch

from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM

# need to implement a 'greedy_decode' method that wraps around calling the LM head of EsmMaskedLM

class EsmWrapper:
    def __init__(self, params):

        self.common_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        esm_tokenizer = EsmTokenizer.from_pretrained(
                    params['esm_path'], 
                    local_files_only=True
                ) 
        # esm_model = EsmModel.from_pretrained(
        #     params['esm_path'], 
        #     local_files_only=True
        # ) 
        esm_lm = EsmForMaskedLM.from_pretrained(
            params['esm_path'], 
            local_files_only=True
        ) 

        self.esm_tokenizer = esm_tokenizer
        self.esm_model   = esm_lm.esm # already has it built-in
        self.esm_lm_head = esm_lm.lm_head

    def encode(self, x):
        """
        Encode a sequence of amino acids into a tensor.
        """
        # Tokenize the input
        _input = self.esm_tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        # Get the embeddings
        _embeddings = self.esm_model(**_input).last_hidden_state
        return _embeddings

    def greedy_decode(self, embeddings):
        """
        Not really 'greedy' decoding in this case. 
        But the BO loop class assumes that the generative model has a 'greedy_decode' method.
        """
        # Get the logits
        _logits = self.esm_lm_head(embeddings)
        # Get the most probable tokens
        _pred_tokens = torch.argmax(_logits, dim=-1)
        
        sequences = []
        for i, _tokenized in enumerate(_pred_tokens.tolist()):
            _out = self.esm_tokenizer.decode(_tokenized)
            _out = _out.split(" ")
            # _out = [c for c in _out if c not in ["<cls>", "<eos>", "<pad>"]]
            _out = [c for c in _out if c in list(self.common_amino_acids)]
            sequences.append("".join(_out))
        
        return sequences
