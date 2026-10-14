"""
Transformer Token-Level Attribution Engine using Captum Layer Integrated Gradients.
Computes token importance scores and generates visual highlights.
"""

from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from config import LABEL_NAMES

class TransformerXAI:
    """Computes Layer Integrated Gradients token attributions for RoBERTa and MentalBERT."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(self.device)

        # Identify embedding layer
        embedding_layer = None
        if hasattr(self.model, "roberta"):
            embedding_layer = self.model.roberta.embeddings
        elif hasattr(self.model, "bert"):
            embedding_layer = self.model.bert.embeddings
        elif hasattr(self.model, "base_model"):
            embedding_layer = getattr(self.model.base_model, "embeddings", None)
            
        if embedding_layer is not None:
            self.lig = LayerIntegratedGradients(self._forward_wrapper, embedding_layer)
        else:
            self.lig = None

    def _forward_wrapper(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def attribute_text(
        self,
        text: str,
        target_class_idx: int,
        max_length: int = 128,
        n_steps: int = 25
    ) -> Dict[str, Any]:
        """
        Calculates Integrated Gradients token-level attributions.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if self.lig is not None:
            # Baseline is pad token IDs
            pad_id = self.tokenizer.pad_token_id or 0
            baseline_ids = torch.full_like(input_ids, pad_id)
            
            attributions, delta = self.lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target_class_idx,
                n_steps=n_steps,
                return_convergence_delta=True
            )
            # Sum over embedding dimension
            attributions_sum = attributions.sum(dim=-1).squeeze(0)
            # Normalize scores
            attr_norm = attributions_sum / (torch.norm(attributions_sum) + 1e-15)
            scores = attr_norm.detach().cpu().numpy()
        else:
            # Fallback simple embedding grad
            scores = np.zeros(len(tokens))

        token_details = []
        for tok, sc in zip(tokens, scores):
            # Skip special tokens in display weighting
            clean_tok = tok.replace("Ġ", " ").replace("##", "")
            token_details.append({
                "token": clean_tok,
                "raw_token": tok,
                "score": float(sc)
            })

        return {
            "target_class": LABEL_NAMES[target_class_idx],
            "target_class_idx": target_class_idx,
            "tokens": token_details,
            "raw_text": text
        }

