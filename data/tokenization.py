"""SMILES tokenizer that reuses the Llama tokenizer plus special tokens."""

from typing import List, Optional


class SMILESTokenizer:
    """Wraps a HuggingFace tokenizer for SMILES string tokenization.

    SMILES strings are treated as regular text and tokenized by the LLM's
    own tokenizer, which typically handles alphanumeric + special chars well.
    """

    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer

    def encode(self, smiles: str, max_length: int = 256) -> List[int]:
        tokens = self.tokenizer.encode(smiles, add_special_tokens=False)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def __len__(self):
        return len(self.tokenizer)
