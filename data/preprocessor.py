"""
Dual Preprocessing Pipelines for Classical ML and Transformer NLP Models.
Preserves linguistic context, negations, and affective punctuation.
"""

import re
import html
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from config import ROBERTA_CHECKPOINT, MENTAL_ROBERTA_CHECKPOINT

# Custom Stopwords that rigorously preserve negations and self-referential markers
NEGATION_WORDS = {"no", "not", "never", "neither", "nor", "none", "nobody", "nowhere", "nothing", "hardly", "barely", "scarcely", "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "doesnt", "wouldnt", "shouldnt", "couldnt"}

def clean_academic_text(text: str) -> str:
    """
    Common Preprocessing:
    - HTML entity unescaping
    - Unicode NFKC normalization
    - Preserves meaningful affective punctuation (! ? ... ')
    - Whitespace normalization
    """
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities
    text = html.unescape(text)
    # Unicode Normalization
    text = unicodedata.normalize("NFKC", text)
    # Remove URL remnants and reddit formatting artifacts
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
    text = re.sub(r'&gt;|&lt;|&amp;', ' ', text)
    # Normalize multiple punctuation while keeping emotional signals
    text = re.sub(r'\.{3,}', ' ... ', text)
    text = re.sub(r'!{2,}', ' !! ', text)
    text = re.sub(r'\?{2,}', ' ?? ', text)
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ClassicalPreprocessor:
    """Configurable TF-IDF feature extractor for Logistic Regression and XGBoost."""
    
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        # Build vectorizer
        from sklearn.feature_extraction import text
        default_stop_words = list(text.ENGLISH_STOP_WORDS - NEGATION_WORDS)
        
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_academic_text,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words=default_stop_words
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]):
        """Fits vectorizer strictly on training texts and returns sparse matrix."""
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X

    def transform(self, texts: List[str]):
        """Transforms validation or test texts."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()


class TransformerPreprocessor:
    """Prepares subword token sequences, attention masks, and tensors for RoBERTa / MentalBERT."""
    
    def __init__(
        self,
        checkpoint: str = ROBERTA_CHECKPOINT,
        max_length: int = 128
    ):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_texts(self, texts: List[str], padding: bool = True, truncation: bool = True):
        """Tokenizes text list and returns PyTorch tensor dictionary."""
        cleaned = [clean_academic_text(t) for t in texts]
        return self.tokenizer(
            cleaned,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def tokenize_single(self, text: str):
        """Tokenizes single text returning token list and encoded dict."""
        cleaned = clean_academic_text(text)
        enc = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return tokens, enc

